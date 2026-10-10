import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO

from env.dual_env import DualKOFEngine

DEFAULT_P1_PATH = "checkpoints_dual/p1_100000"
DEFAULT_P2_PATH = "checkpoints_dual/p2_100000"
NUM_EPISODES = 5
ACTION_COUNT = 17  # must match GameController.action_count() / make_spaces()


class RandomPolicy:
    def __init__(self, n_actions: int = ACTION_COUNT, seed: int | None = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = False):
        return int(self.rng.integers(self.n_actions)), None


def load_policy(path: str, label: str):
    if path.lower() == "random":
        print(f"{label}: random baseline")
        return RandomPolicy()

    if not os.path.exists(path + ".zip") and not os.path.exists(path):
        raise FileNotFoundError(f"{label}: model not found at '{path}.zip'")

    print(f"{label}: loading {path}.zip")

    return PPO.load(path)


def _to_int(action) -> int:
    """PPO.predict returns a 0-d/1-element array; RandomPolicy returns an int."""
    return int(np.asarray(action).reshape(-1)[0])


def decide_winner(info: dict, mode: str, terminated: bool, dmg_p1: float, dmg_p2: float) -> str:
    if mode != "match":
        return "n/a"
    p1b, p2b = info["p1_bars"], info["p2_bars"]
    p1h, p2h = info["p1_hp"], info["p2_hp"]

    if terminated and p1b <= 0 and p2b <= 0:
        return "ambiguous"
    if p1b != p2b:
        return "P1" if p1b > p2b else "P2"
    if terminated:
        return "draw"
    if abs(p1h - p2h) > 1e-6:
        return "P1" if p1h > p2h else "P2"
    return "draw"


def run_dual(p1_path: str, p2_path: str, mode: str, episodes: int, deterministic: bool) -> None:
    model_p1 = load_policy(p1_path, "P1")
    model_p2 = load_policy(p2_path, "P2")

    print(f"Starting DualKOFEngine (mode={mode}). Focus the game window now...")
    engine = DualKOFEngine(mode=mode)

    wins = {"P1": 0, "P2": 0, "draw": 0, "ambiguous": 0}
    dmg_edge = {"P1": 0, "P2": 0, "tie": 0}
    totals = {"p1_reward": [], "p2_reward": [], "p1_damage": [], "p2_damage": [], "steps": []}

    try:
        for ep in range(1, episodes + 1):
            obs_p1, obs_p2 = engine.reset()
            reward_p1 = reward_p2 = 0.0
            damage_p1 = damage_p2 = 0.0
            steps = 0
            done = False
            info: dict = {}

            print(f"\n── Episode {ep}/{episodes} ──")

            while not done:
                # Both agents decide from the same tick's observation...
                action_p1, _ = model_p1.predict(obs_p1, deterministic=deterministic)
                action_p2, _ = model_p2.predict(obs_p2, deterministic=deterministic)

                # ...and both actions go into the game on the same tick.
                obs_p1, obs_p2, r1, r2, terminated, truncated, info = engine.step(
                    _to_int(action_p1), _to_int(action_p2)
                )
                reward_p1 += r1
                reward_p2 += r2
                damage_p1 += info["p1_damage_dealt"]
                damage_p2 += info["p2_damage_dealt"]
                steps += 1
                done = terminated or truncated

            winner = decide_winner(info, mode, terminated, damage_p1, damage_p2)
            if winner in wins:
                wins[winner] += 1
            if abs(damage_p1 - damage_p2) < 1e-6:
                dmg_edge["tie"] += 1
            else:
                dmg_edge["P1" if damage_p1 > damage_p2 else "P2"] += 1

            totals["p1_reward"].append(reward_p1)
            totals["p2_reward"].append(reward_p2)
            totals["p1_damage"].append(damage_p1)
            totals["p2_damage"].append(damage_p2)
            totals["steps"].append(steps)

            if winner in ("P1", "P2"):
                outcome = f"  → {winner} wins"
            elif winner == "draw":
                outcome = "  → DRAW"
            elif winner == "ambiguous":
                outcome = "  → AMBIGUOUS (both bars hit 0 — see note)"
            else:
                outcome = ""
            print(
                f"Steps: {steps}  |  "
                f"P1 reward: {reward_p1:.2f}  P2 reward: {reward_p2:.2f}  |  "
                f"P1 bars: {info['p1_bars']}  P2 bars: {info['p2_bars']}  |  "
                f"dmg dealt P1: {damage_p1:.2f}  P2: {damage_p2:.2f}{outcome}"
            )

            # brief pause so the game can settle before the next reset
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nInterrupted — summarising the episodes completed so far.")
    finally:
        engine.close()

    played = len(totals["steps"])
    if played == 0:
        print("No episodes completed.")
        return

    print("\n══════════ Summary ══════════")
    print(f"Episodes played : {played}")
    if mode == "match":
        print(f"P1 wins         : {wins['P1']}  ({wins['P1'] / played:.0%})")
        print(f"P2 wins         : {wins['P2']}  ({wins['P2'] / played:.0%})")
        print(f"Draws           : {wins['draw']}")
        print(f"Ambiguous       : {wins['ambiguous']}  (both bars hit 0 in the same episode)")
        print(f"More damage     : P1 {dmg_edge['P1']}  |  P2 {dmg_edge['P2']}  |  tie {dmg_edge['tie']}")
    print(f"Avg reward      : P1 {np.mean(totals['p1_reward']):.2f}  |  P2 {np.mean(totals['p2_reward']):.2f}")
    print(f"Avg damage dealt: P1 {np.mean(totals['p1_damage']):.2f}  |  P2 {np.mean(totals['p2_damage']):.2f}")
    print(f"Avg steps       : {np.mean(totals['steps']):.0f}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pit two trained KOF RL agents against each other.")
    parser.add_argument("--p1", type=str, default=DEFAULT_P1_PATH,
                        help="P1 model path without .zip, or 'random' (default: %(default)s)")
    parser.add_argument("--p2", type=str, default=DEFAULT_P2_PATH,
                        help="P2 model path without .zip, or 'random' (default: %(default)s)")
    parser.add_argument("--mode", type=str, default="match", choices=["training", "match"],
                        help="Engine mode (default: match — real 2-bar rules and a winner)")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES,
                        help="Number of episodes to play (default: %(default)s)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Always pick the most likely action (default: sample)")
    args = parser.parse_args()

    run_dual(args.p1, args.p2, args.mode, args.episodes, args.deterministic)
