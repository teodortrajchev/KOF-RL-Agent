import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass

from env.controls import GameController
from env.vision import ScreenCapture
from env.position import PositionTracker


# GAME STATE
@dataclass
class GameState:
    player_hp: float = 1.0
    enemy_hp: float = 1.0


# ENVIRONMENT
class KOFEnv(gym.Env):


    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    MAX_STEPS = 750  # ~1 minute at 0.08s step
    STACK_SIZE = 4
    VECTOR_DIM = 7  # [dx_norm, buff_onehot(3), time_remaining_norm, hp_diff, enemy_hp]

    # life-lost (bar-loss) detection tuning
    LOW_HP_THRESHOLD = 0.15
    HIGH_HP_THRESHOLD = 0.8
    LOW_STREAK_REQUIRED = 2
    RESET_GRACE_STEPS = 15

    def __init__(self, game_region=None, render_mode=None, mode="training"):
        super().__init__()

        assert mode in ("training", "match"), f"unknown mode: {mode}"
        self.mode = mode
        self.render_mode = render_mode

        # core modules
        self.controller = GameController()
        self.capture = ScreenCapture(game_region=game_region)
        self.position_tracker = PositionTracker()

        #match state
        self.player_bars = 2
        self.enemy_bars = 2

        #frame stacking
        self.stack_size = self.STACK_SIZE
        self.frame_stack = []

        #action space
        self.action_space = spaces.Discrete(self.controller.action_count())


        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(84, 84, self.stack_size),
                dtype=np.uint8,
            ),
            "vector": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.VECTOR_DIM,),
                dtype=np.float32,
            ),
        })

        # internal state
        self._step_count = 0
        self._current_obs = None
        self._prev_state = GameState()
        self._enemy_is_right = True  # updated each step from position tracker

        self._damage_dealt_buffer = 0.0
        self._damage_taken_buffer = 0.0
        self._enemy_low_streak = 0
        self._player_low_streak = 0

    # RESET
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.controller.release_all()
        self.position_tracker.reset()
        time.sleep(1.0)


        if self.mode == "match":
            self._ensure_round_started()

        self._step_count = 0
        self.player_bars = 2
        self.enemy_bars = 2
        self._damage_dealt_buffer = 0.0
        self._damage_taken_buffer = 0.0
        self._enemy_is_right = True
        self._enemy_low_streak = 0
        self._player_low_streak = 0

        frame, player_hp, enemy_hp, raw_bgr = self.capture.get_observation_bundle()
        self.position_tracker.update(raw_bgr)

        self.frame_stack = [frame.copy() for _ in range(self.stack_size)]
        self._prev_state = GameState(player_hp=player_hp, enemy_hp=enemy_hp)

        image_obs = np.concatenate(self.frame_stack, axis=2)
        vector_obs = self._build_vector_obs(player_hp, enemy_hp)

        obs = {"image": image_obs, "vector": vector_obs}
        self._current_obs = obs

        return obs, {"step": 0}

    # MATCH-MODE ROUND-START
    def _ensure_round_started(
        self,
        timeout: float = 20.0,
        poll_interval: float = 1.2,
        confirm_keys: tuple = ("enter", "space", "u"),
    ):
        start = time.time()
        key_idx = 0

        while time.time() - start < timeout:
            _, player_hp, enemy_hp, _ = self.capture.get_observation_bundle()

            if player_hp > 0.05 and enemy_hp > 0.05:
                return True

            key = confirm_keys[key_idx % len(confirm_keys)]
            self.controller.tap_key(key)
            key_idx += 1
            time.sleep(poll_interval)

        print(
            "[KOFEnv] WARNING: could not confirm round start within "
            f"{timeout:.0f}s — proceeding anyway. If this keeps happening, "
            "check the confirm_keys list in _ensure_round_started()."
        )
        return False

    # STACKING
    def _stack(self, frame):
        self.frame_stack.append(frame)
        if len(self.frame_stack) > self.stack_size:
            self.frame_stack.pop(0)
        return np.concatenate(self.frame_stack, axis=2)

    # VECTOR OBS
    def _build_vector_obs(self, player_hp, enemy_hp):
        dx_norm = self._last_dx_norm()

        buff_onehot = [0.0, 0.0, 0.0]
        if self.controller.buff_state in (1, 2, 3):
            buff_onehot[self.controller.buff_state - 1] = 1.0

        time_remaining_norm = 1.0 - (self._step_count / self.MAX_STEPS)
        hp_diff = player_hp - enemy_hp  # negative = you're losing

        vec = np.array(
            [dx_norm, *buff_onehot, time_remaining_norm, hp_diff, enemy_hp],
            dtype=np.float32,
        )
        return np.clip(vec, -1.0, 1.0)

    def _last_dx_norm(self):
        return self._last_pos["dx_norm"] if getattr(self, "_last_pos", None) else 0.0

    # STEP
    def step(self, action: int):

        old_player_hp = self._prev_state.player_hp
        old_enemy_hp = self._prev_state.enemy_hp

        self.controller.execute_action(action, enemy_is_right=self._enemy_is_right)

        time.sleep(0.08)

        # single capture for this step — fixes the old 3x-grab desync bug
        frame, player_hp, enemy_hp, raw_bgr = self.capture.get_observation_bundle()
        image_obs = self._stack(frame)

        pos_result = self.position_tracker.update(raw_bgr)
        self._last_pos=pos_result
        self._enemy_is_right = pos_result["enemy_is_right"]

        if self._step_count < self.RESET_GRACE_STEPS:
            enemy_life_lost = False
            player_life_lost = False
            self._enemy_low_streak = 0
            self._player_low_streak = 0
        else:
            self._enemy_low_streak = (
                self._enemy_low_streak + 1 if old_enemy_hp < self.LOW_HP_THRESHOLD else 0
            )
            self._player_low_streak = (
                self._player_low_streak + 1 if old_player_hp < self.LOW_HP_THRESHOLD else 0
            )

            enemy_life_lost = (
                self._enemy_low_streak >= self.LOW_STREAK_REQUIRED
                and enemy_hp > self.HIGH_HP_THRESHOLD
            )
            player_life_lost = (
                self._player_low_streak >= self.LOW_STREAK_REQUIRED
                and player_hp > self.HIGH_HP_THRESHOLD
            )

            if enemy_life_lost:
                self._enemy_low_streak = 0
            if player_life_lost:
                self._player_low_streak = 0

        if enemy_life_lost:
            enemy_hp_for_state = old_enemy_hp
        else:
            enemy_hp_for_state = enemy_hp

        if player_life_lost:
            player_hp_for_state = old_player_hp
        else:
            player_hp_for_state = player_hp

        self._prev_state = GameState(player_hp_for_state, enemy_hp_for_state)

        enemy_bar_lost = False
        player_bar_lost = False

        if self.mode == "match":
            if enemy_life_lost:
                self.enemy_bars -= 1
                enemy_bar_lost = True
            if player_life_lost:
                self.player_bars -= 1
                player_bar_lost = True

        # DAMAGE THIS STEP
        damage_dealt = max(0.0, old_enemy_hp - enemy_hp) if not enemy_life_lost else 0.0
        damage_taken = max(0.0, old_player_hp - player_hp) if not player_life_lost else 0.0

        reward_damage = 0.0
        reward_shaping = 0.0
        reward_terminal = 0.0


        reward_damage += damage_dealt * 15.0
        reward_damage -= damage_taken * 12.0


        if damage_dealt > 0 and enemy_hp < 0.25:
            reward_damage += damage_dealt * 10.0

        # idling / retreating when the enemyis nearly dead should be discouraged.
        PASSIVE_ACTIONS = {0, 4, 10, 14}  # idle, crouch, block, backdash
        if enemy_hp < 0.15 and action in PASSIVE_ACTIONS:
            reward_shaping -= 0.15

        # survival pressure
        reward_shaping += 0.001

        # generic anti-idle
        if action == 0:
            reward_shaping -= 0.02

        # life-lost reward
        if enemy_life_lost:
            reward_terminal += 5.0
        if player_life_lost:
            reward_terminal -= 5.0

        # time pressure nudge
        reward_shaping += (1.0 - self._step_count / self.MAX_STEPS) * 0.01

        self._step_count += 1

        # TERMINATION
        terminated = False
        if self.mode == "match":
            terminated = self.enemy_bars <= 0 or self.player_bars <= 0
            if self.enemy_bars <= 0:
                reward_terminal += 100.0
            if self.player_bars <= 0:
                reward_terminal -= 100.0

        truncated = self._step_count >= self.MAX_STEPS


        if truncated and self.mode == "match" and not terminated:
            if self.player_bars != self.enemy_bars:
                if self.player_bars > self.enemy_bars:
                    reward_terminal += 100.0
                else:
                    reward_terminal -= 100.0
            else:
                if player_hp > enemy_hp + 1e-6:
                    reward_terminal += 100.0
                elif enemy_hp > player_hp + 1e-6:
                    reward_terminal -= 100.0
                # equal HP → draw → no bonus either way

        # scale all three buckets by the same /10 factor so they stay directly comparable to the total reward and to each other
        reward_damage /= 10.0
        reward_shaping /= 10.0
        reward_terminal /= 10.0
        reward = reward_damage + reward_shaping + reward_terminal

        info = {
            "step": self._step_count,
            "action": action,
            "reward": reward,
            "reward_damage": reward_damage,
            "reward_shaping": reward_shaping,
            "reward_terminal": reward_terminal,
            "player_hp": player_hp,
            "enemy_hp": enemy_hp,
            "player_bars": self.player_bars,
            "enemy_bars": self.enemy_bars,
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "enemy_is_right": self._enemy_is_right,
            "position_valid": pos_result["valid"],
            "buff_state": self.controller.buff_state,
        }

        vector_obs = self._build_vector_obs(player_hp, enemy_hp)
        obs = {"image": image_obs, "vector": vector_obs}
        self._current_obs = obs

        return obs, reward, terminated, truncated, info

    # RENDER
    def render(self):
        if self._current_obs is None:
            return None

        image = self._current_obs["image"]

        if self.render_mode == "rgb_array":
            g = image[:, :, -1]
            return np.stack([g, g, g], axis=-1)

        if self.render_mode == "human":
            import cv2
            g = image[:, :, -1]
            g = cv2.resize(g, (420, 420), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("KOFEnv v5", g)
            cv2.waitKey(1)

        return None

    # CLOSE
    def close(self):
        self.controller.release_all()
        self.capture.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass