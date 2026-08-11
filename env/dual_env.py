# env/dual_env.py
#
# PURPOSE:
#   Drive P1 and P2 with two independent agents against each other in the
#   game's real local-2P mode, on ONE shared game tick per step — no
#   threading, no synchronized processes. Each call to `DualKOFEngine.step()`
#   sends both players' keys, sleeps once, captures the screen once, and
#   returns a separate (obs, reward, ...) tuple for each side.
#
#   This is NOT a gym.Env — it can't be, since a gym.Env.step() only takes
#   one action. Drive it from a custom training loop (see train_dual.py)
#   that calls DualKOFEngine.step(action_p1, action_p2) once per tick and
#   feeds the two returned experiences into two separate PPO models.
#
# REUSES:
#   - env/vision.py's ScreenCapture as-is: PLAYER_HP_REGION is P1's bar,
#     ENEMY_HP_REGION is P2's bar (already fixed UI positions).
#   - env/position.py's PositionTracker as-is: with assume_self_starts_left
#     =True, "self" == P1's blob, "enemy" == P2's blob. P2's perspective is
#     just the sign-flip / negation of P1's result — no second tracker
#     needed.
#   - The life-lost detection and reward-shaping logic from KOFEnv.step(),
#     factored into a helper (`_compute_reward`) so it can be called once
#     per side without duplicating the tuning constants.

import time
import numpy as np
from dataclasses import dataclass

from env.controls_2p import *
from env.vision import ScreenCapture
from env.position import PositionTracker


@dataclass
class DualGameState:
    p1_hp: float = 1.0
    p2_hp: float = 1.0


class DualKOFEngine:
    """
    mode="training": infinite lives, episode ends only by MAX_STEPS.
    mode="match": real 2-bar system per side, terminates when either bar
                  hits 0, with the same timeout tie-break rule as KOFEnv.
    """

    MAX_STEPS = 750
    STACK_SIZE = 4
    VECTOR_DIM = 7  # [dx_norm, buff_onehot(3), time_remaining_norm, hp_diff, opp_hp]

    LOW_HP_THRESHOLD = 0.15
    HIGH_HP_THRESHOLD = 0.8
    LOW_STREAK_REQUIRED = 2
    RESET_GRACE_STEPS = 15

    def __init__(self, game_region=None, mode="training"):
        assert mode in ("training", "match"), f"unknown mode: {mode}"
        self.mode = mode

        self.p1_controller = GameController(keymap=P1_KEYMAP)
        self.p2_controller = GameController(keymap=P2_KEYMAP)
        self.capture = ScreenCapture(game_region=game_region)
        self.position_tracker = PositionTracker(assume_self_starts_left=True)

        self.p1_bars = 2
        self.p2_bars = 2

        self.stack_size = self.STACK_SIZE
        self.frame_stack = []

        self._step_count = 0
        self._prev_state = DualGameState()
        self._p1_enemy_is_right = True  # is P2 to P1's right?
        self._p2_enemy_is_right = True  # is P1 to P2's right?

        self._p1_low_streak = 0
        self._p2_low_streak = 0

    # ─────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────
    def reset(self):
        self.p1_controller.release_all()
        self.p2_controller.release_all()
        self.position_tracker.reset()
        time.sleep(1.0)

        if self.mode == "match":
            self._ensure_round_started()

        self._step_count = 0
        self.p1_bars = 2
        self.p2_bars = 2
        self._p1_enemy_is_right = True
        self._p2_enemy_is_right = True
        self._p1_low_streak = 0
        self._p2_low_streak = 0

        frame, p1_hp, p2_hp, raw_bgr = self.capture.get_observation_bundle()
        self.position_tracker.update(raw_bgr)

        self.frame_stack = [frame.copy() for _ in range(self.stack_size)]
        self._prev_state = DualGameState(p1_hp=p1_hp, p2_hp=p2_hp)

        image_obs = np.concatenate(self.frame_stack, axis=2)
        obs_p1 = {
            "image": image_obs,
            "vector": self._build_vector(
                own_hp=p1_hp, opp_hp=p2_hp, dx_norm=0.0,
                buff_state=self.p1_controller.buff_state,
            ),
        }
        obs_p2 = {
            "image": image_obs,
            "vector": self._build_vector(
                own_hp=p2_hp, opp_hp=p1_hp, dx_norm=0.0,
                buff_state=self.p2_controller.buff_state,
            ),
        }
        return obs_p1, obs_p2

    def _ensure_round_started(
        self,
        timeout: float = 20.0,
        poll_interval: float = 1.2,
        confirm_keys: tuple = ("enter", "space"),
    ):
        start = time.time()
        key_idx = 0
        while time.time() - start < timeout:
            _, p1_hp, p2_hp, _ = self.capture.get_observation_bundle()
            if p1_hp > 0.05 and p2_hp > 0.05:
                return True
            key = confirm_keys[key_idx % len(confirm_keys)]
            self.p1_controller.tap_key(key)
            key_idx += 1
            time.sleep(poll_interval)
        print(
            "[DualKOFEngine] WARNING: could not confirm round start within "
            f"{timeout:.0f}s — proceeding anyway."
        )
        return False

    # ─────────────────────────────────────────────
    # VECTOR OBS
    # ─────────────────────────────────────────────
    def _build_vector(self, own_hp, opp_hp, dx_norm, buff_state):
        buff_onehot = [0.0, 0.0, 0.0]
        if buff_state in (1, 2, 3):
            buff_onehot[buff_state - 1] = 1.0

        time_remaining_norm = 1.0 - (self._step_count / self.MAX_STEPS)
        hp_diff = own_hp - opp_hp  # positive = you're winning

        vec = np.array(
            [dx_norm, *buff_onehot, time_remaining_norm, hp_diff, opp_hp],
            dtype=np.float32,
        )
        return np.clip(vec, -1.0, 1.0)

    def _stack(self, frame):
        self.frame_stack.append(frame)
        if len(self.frame_stack) > self.stack_size:
            self.frame_stack.pop(0)
        return np.concatenate(self.frame_stack, axis=2)

    # ─────────────────────────────────────────────
    # REWARD (identical tuning to KOFEnv.step(), called once per side)
    # ─────────────────────────────────────────────
    def _compute_reward(
        self, own_hp, opp_hp, old_own_hp, old_opp_hp,
        own_life_lost, opp_life_lost, action, own_bars, opp_bars,
    ):
        damage_dealt = max(0.0, old_opp_hp - opp_hp) if not opp_life_lost else 0.0
        damage_taken = max(0.0, old_own_hp - own_hp) if not own_life_lost else 0.0

        reward_damage = damage_dealt * 15.0 - damage_taken * 12.0

        if damage_dealt > 0 and opp_hp < 0.25:
            reward_damage += damage_dealt * 10.0

        reward_shaping = 0.0
        PASSIVE_ACTIONS = {0, 4, 10, 14}
        if opp_hp < 0.15 and action in PASSIVE_ACTIONS:
            reward_shaping -= 0.15

        reward_shaping += 0.001
        if action == 0:
            reward_shaping -= 0.02
        reward_shaping += (1.0 - self._step_count / self.MAX_STEPS) * 0.01

        reward_terminal = 0.0
        if opp_life_lost:
            reward_terminal += 5.0
        if own_life_lost:
            reward_terminal -= 5.0

        terminated = False
        if self.mode == "match":
            terminated = own_bars <= 0 or opp_bars <= 0
            if opp_bars <= 0:
                reward_terminal += 100.0
            if own_bars <= 0:
                reward_terminal -= 100.0

        reward = (reward_damage + reward_shaping + reward_terminal) / 10.0
        return reward, terminated, damage_dealt, damage_taken

    # ─────────────────────────────────────────────
    # STEP — one shared tick, two independent results
    # ─────────────────────────────────────────────
    def step(self, action_p1: int, action_p2: int):
        old_p1_hp = self._prev_state.p1_hp
        old_p2_hp = self._prev_state.p2_hp

        # Send both players' inputs within the same tick.
        self.p1_controller.execute_action(action_p1, enemy_is_right=self._p1_enemy_is_right)
        self.p2_controller.execute_action(action_p2, enemy_is_right=self._p2_enemy_is_right)

        time.sleep(0.08)

        frame, p1_hp, p2_hp, raw_bgr = self.capture.get_observation_bundle()
        image_obs = self._stack(frame)

        pos_result = self.position_tracker.update(raw_bgr)
        self._p1_enemy_is_right = pos_result["enemy_is_right"]
        self._p2_enemy_is_right = (not pos_result["enemy_is_right"]) if pos_result["valid"] else True
        p1_dx_norm = pos_result["dx_norm"]
        p2_dx_norm = -p1_dx_norm

        # ── life-lost detection (mirrors KOFEnv, run for both sides) ──
        if self._step_count < self.RESET_GRACE_STEPS:
            p1_life_lost = False
            p2_life_lost = False
            self._p1_low_streak = 0
            self._p2_low_streak = 0
        else:
            self._p1_low_streak = self._p1_low_streak + 1 if old_p1_hp < self.LOW_HP_THRESHOLD else 0
            self._p2_low_streak = self._p2_low_streak + 1 if old_p2_hp < self.LOW_HP_THRESHOLD else 0

            p1_life_lost = (
                self._p1_low_streak >= self.LOW_STREAK_REQUIRED and p1_hp > self.HIGH_HP_THRESHOLD
            )
            p2_life_lost = (
                self._p2_low_streak >= self.LOW_STREAK_REQUIRED and p2_hp > self.HIGH_HP_THRESHOLD
            )
            if p1_life_lost:
                self._p1_low_streak = 0
            if p2_life_lost:
                self._p2_low_streak = 0

        p1_hp_for_state = old_p1_hp if p1_life_lost else p1_hp
        p2_hp_for_state = old_p2_hp if p2_life_lost else p2_hp
        self._prev_state = DualGameState(p1_hp_for_state, p2_hp_for_state)

        if self.mode == "match":
            if p1_life_lost:
                self.p1_bars -= 1
            if p2_life_lost:
                self.p2_bars -= 1

        self._step_count += 1

        reward_p1, term_p1, dmg_p1_dealt, dmg_p1_taken = self._compute_reward(
            own_hp=p1_hp, opp_hp=p2_hp, old_own_hp=old_p1_hp, old_opp_hp=old_p2_hp,
            own_life_lost=p1_life_lost, opp_life_lost=p2_life_lost,
            action=action_p1, own_bars=self.p1_bars, opp_bars=self.p2_bars,
        )
        reward_p2, term_p2, dmg_p2_dealt, dmg_p2_taken = self._compute_reward(
            own_hp=p2_hp, opp_hp=p1_hp, old_own_hp=old_p2_hp, old_opp_hp=old_p1_hp,
            own_life_lost=p2_life_lost, opp_life_lost=p1_life_lost,
            action=action_p2, own_bars=self.p2_bars, opp_bars=self.p1_bars,
        )
        terminated = term_p1 or term_p2  # one shared episode boundary

        truncated = self._step_count >= self.MAX_STEPS
        if truncated and self.mode == "match" and not terminated:
            if self.p1_bars != self.p2_bars:
                bonus = 100.0 if self.p1_bars > self.p2_bars else -100.0
            elif p1_hp > p2_hp + 1e-6:
                bonus = 100.0
            elif p2_hp > p1_hp + 1e-6:
                bonus = -100.0
            else:
                bonus = 0.0
            reward_p1 += bonus / 10.0
            reward_p2 -= bonus / 10.0

        obs_p1 = {
            "image": image_obs,
            "vector": self._build_vector(p1_hp, p2_hp, p1_dx_norm, self.p1_controller.buff_state),
        }
        obs_p2 = {
            "image": image_obs,
            "vector": self._build_vector(p2_hp, p1_hp, p2_dx_norm, self.p2_controller.buff_state),
        }

        info = {
            "step": self._step_count,
            "p1_hp": p1_hp, "p2_hp": p2_hp,
            "p1_bars": self.p1_bars, "p2_bars": self.p2_bars,
            "p1_damage_dealt": dmg_p1_dealt, "p2_damage_dealt": dmg_p2_dealt,
            "position_valid": pos_result["valid"],
        }

        return obs_p1, obs_p2, reward_p1, reward_p2, terminated, truncated, info

    def close(self):
        self.p1_controller.release_all()
        self.p2_controller.release_all()
        self.capture.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass