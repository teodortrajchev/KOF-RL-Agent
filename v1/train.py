import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from v1.kof_env import KOFEnv


# PATHS
LOG_DIR = "../logs"
CHECKPOINT_DIR = "../checkpoints"
BEST_MODEL_DIR = "../best_model"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


MATCH_EVERY_N_EPISODES = 5


class ModeCurriculumCallback(BaseCallback):
    """
    Switches the training env's mode between "training" and "match"
    based on episode count. Every Nth completed episode is followed by
    one match-mode episode.All others run in training mode.
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
            target_env = self.training_env.envs[0].unwrapped

            if self.episode_count % self.match_every_n == 0:
                if target_env.mode != "match":
                    target_env.mode = "match"
                    print(f"[curriculum] episode {self.episode_count}: -> match mode")
            else:
                if target_env.mode != "training":
                    target_env.mode = "training"
                    print(f"[curriculum] episode {self.episode_count}: -> training mode")

        return True


def make_env(mode: str):
    def _init():
        return Monitor(KOFEnv(render_mode=None, mode=mode))

    return _init

TOTAL_TIMESTEPS = 1_000_000

def train():

    print("Creating vectorized environment (curriculum: training + periodic match) …")


    env = DummyVecEnv([make_env("training")])


    # MODEL — MultiInputPolicy, because observation_space is now a Dict
    # ({"image": ..., "vector": ...}). SB3's CombinedExtractor runs a
    # NatureCNN over "image" and an MLP over "vector", then concatenates
    # — this is how side-detection / buff-state / time-remaining get to
    # the policy without polluting the pixel input.
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

    # CALLBACKS
    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="kof_ppo"
    )



    curriculum_cb = ModeCurriculumCallback()

    # TRAIN
    print("Starting PPO training …")
    print("Open TensorBoard: tensorboard --logdir ./logs")
    print(
        f"Curriculum: training mode by default, match mode every "
        f"{MATCH_EVERY_N_EPISODES} episodes."
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, curriculum_cb],
        reset_num_timesteps=True
    )

    # SAVE FINAL MODEL
    model.save(os.path.join(CHECKPOINT_DIR, "kof_ppo_final"))

    print("\nTraining complete. Model saved.")

if __name__ == "__main__":
    train()