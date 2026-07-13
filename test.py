#   Load a saved PPO model and run it in the game without any training.
#   Use this to evaluate how well the agent plays after training.
#   python test.py --model checkpoints/kof_ppo_final
#   python test.py --mode match
#   python test.py

import argparse
from stable_baselines3 import PPO
from env.kof_env import KOFEnv

DEFAULT_MODEL_PATH = "checkpoints/kof_ppo_20000_steps"
NUM_EPISODES= 5
RENDER= True  #False to disable cv2 preview window

def run_agent(model_path: str, mode: str) -> None:
    print(f"Loading model: {model_path}.zip  (eval mode={mode})")
    env   = KOFEnv(render_mode="human" if RENDER else None, mode=mode)
    model = PPO.load(model_path, env=env)

    for ep in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps= 0
        done= False

        print(f"\n── Episode {ep}/{NUM_EPISODES} ──")

        while not done:
            # deterministic=True → always pick the most likely action
            # (no exploration noise during evaluation)
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps+= 1
            done = terminated or truncated

            if RENDER:
                env.render()

        outcome = ""
        if mode == "match":
            if info["player_bars"] > info["enemy_bars"]:
                outcome = "  → WIN"
            elif info["player_bars"] < info["enemy_bars"]:
                outcome = "  → LOSS"
            else:
                outcome = "  → DRAW (bars) / decided by HP"

        print(f"Steps: {steps}  |  Total reward: {total_reward:.2f}{outcome}")

    env.close()
    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained KOF RL agent.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to saved .zip model (without the .zip extension)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="match",
        choices=["training", "match"],
        help="Which env mode to evaluate in (default: match, since that's the real target)",
    )
    args = parser.parse_args()
    run_agent(args.model, args.mode)