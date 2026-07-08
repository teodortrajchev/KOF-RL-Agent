import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

from env.kof_env import KOFEnv


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
LOG_DIR = "./logs"
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_DIR = "./best_model"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# MODE CURRICULUMwjaadjsidj and the round keeps going).
# Match mode is the real deployment target but is expensive per episode
# (menu navigation, round-end animations, occasional stuck-on-menu retries).
# For an hours-long unattended run: spend most episodes in training mode,
# but periodically run one match-mode episode so the policy still
# experiences real bar-loss termination and the timeout tie-break rule.
MATCH_EVERY_N_EPISODES = 5


class ModeCurriculumCallback(BaseCallback):
    """
    Switches the training env's mode between "training" and "match"
    based on episode count. Every Nth completed episode is followed by
    one match-mode episode; all others run in training mode.

    Note: SB3's DummyVecEnv auto-resets internally on the same step()
    call that produces done=True, so by the time this callback observes
    the episode boundary, that auto-reset already happened under the OLD
    mode. The mode switch therefore takes effect one episode later than
    it's "counted" here. Harmless over an hours-long run — this is just
    controlling experience mix, not anything safety-critical.
    """

    def __init__(self, match_every_n: int = MATCH_EVERY_N_EPISODES, verbose: int = 0):
        super().__init__(verbose)
        self.match_every_n = match_every_n
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for done in dones:
            if not done:
                continue
            self.episode_count += 1
            target_env = self.training_env.envs[0]

            if self.episode_count % self.match_every_n == 0:
                if target_env.mode != "match":
                    target_env.mode = "match"
                    print(f"[curriculum] episode {self.episode_count}: -> match mode")
            else:
                if target_env.mode != "training":
                    target_env.mode = "training"
                    print(f"[curriculum] episode {self.episode_count}: -> training mode")

        return True


# ─────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────
def make_env(mode: str):
    def _init():
        return KOFEnv(render_mode=None, mode=mode)
    return _init


# ─────────────────────────────────────────────
# TRAIN CONFIG
# ─────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000


def train():

    print("Creating vectorized environment (curriculum: training + periodic match) …")

    # IMPORTANT: KOFEnv already returns a 4-frame-stacked image internally
    # (see STACK_SIZE in kof_env.py). Do NOT also wrap this in
    # VecFrameStack — the previous version did, which stacked already-
    # stacked frames into 16 channels of overlapping, duplicated pixel
    # data. Frame stacking happens in exactly one place: inside the env.
    #
    # Training env starts in "training" mode; the curriculum callback
    # flips it to "match" periodically. Eval env is pinned to "match"
    # always, since that's the real target you want checkpoint quality
    # judged against.
    env = DummyVecEnv([make_env("training")])
    eval_env = DummyVecEnv([make_env("match")])

    # ─────────────────────────────────────────────
    # MODEL — MultiInputPolicy, because observation_space is now a Dict
    # ({"image": ..., "vector": ...}). SB3's CombinedExtractor runs a
    # NatureCNN over "image" and an MLP over "vector", then concatenates
    # — this is how side-detection / buff-state / time-remaining get to
    # the policy without polluting the pixel input.
    # ─────────────────────────────────────────────
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=LOG_DIR,
    )

    # ─────────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="kof_ppo"
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=50_000,
        deterministic=True,
        render=False
    )

    curriculum_cb = ModeCurriculumCallback()

    # ─────────────────────────────────────────────
    # TRAIN
    # ─────────────────────────────────────────────
    print("Starting PPO training …")
    print("Open TensorBoard: tensorboard --logdir ./logs")
    print(
        f"Curriculum: training mode by default, match mode every "
        f"{MATCH_EVERY_N_EPISODES} episodes."
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb, curriculum_cb],
        reset_num_timesteps=True
    )

    # ─────────────────────────────────────────────
    # SAVE FINAL MODEL
    # ─────────────────────────────────────────────
    model.save(os.path.join(CHECKPOINT_DIR, "kof_ppo_final"))

    print("\nTraining complete. Model saved.")


if __name__ == "__main__":
    train()