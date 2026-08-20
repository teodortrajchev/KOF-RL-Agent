# train_dual.py
#
# Trains TWO independent PPO agents — one controlling P1, one controlling
# P2 — against each other in the game's real local-2P mode, on one shared
# game tick per step.
#
# WHY NOT model.learn()?
#   SB3's normal training loop assumes exactly one policy stepping exactly
#   one env. Here, one game tick has to carry BOTH players' actions at
#   once (they're on the same keyboard, same screen, same frame). So this
#   script drives SB3's internals directly instead of calling .learn():
#     1. ask each policy for an action from its own observation
#     2. send both actions to DualKOFEngine.step() in the same tick
#     3. feed each policy's own (obs, action, reward) into its OWN
#        rollout buffer
#     4. once a buffer fills up (n_steps), run that model's normal PPO
#        update (model.train()) — independently for each side
#
# This is intentionally close to what OnPolicyAlgorithm.collect_rollouts()
# does internally in SB3 — it's just doing it for two policies in lockstep
# instead of one. Because it touches SB3 internals (rollout_buffer,
# policy.forward, obs_as_tensor) rather than the public .learn() API, it's
# more fragile across SB3 versions than normal training — if you upgrade
# stable-baselines3 and this breaks, that's the likely reason.

import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import obs_as_tensor, configure_logger

from env.dual_env import DualKOFEngine

# SPACES (shared shape for both agents)

def make_spaces():
    obs_space = spaces.Dict({
        "image": spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8),
        "vector": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
    })
    act_space = spaces.Discrete(17)
    return obs_space, act_space


class _ShellEnv(gym.Env):
    """Dummy single-step env used ONLY to satisfy PPO's constructor (it
    needs an env to read observation_space/action_space from and to set up
    the rollout buffer / policy network shapes). Its step()/reset() are
    never actually called during training — DualKOFEngine drives the real
    game instead."""

    def __init__(self):
        super().__init__()
        self.observation_space, self.action_space = make_spaces()

    def reset(self, *, seed=None, options=None):
        obs = {
            "image": np.zeros((84, 84, 4), dtype=np.uint8),
            "vector": np.zeros((7,), dtype=np.float32),
        }
        return obs, {}

    def step(self, action):
        obs, _ = self.reset()
        return obs, 0.0, False, False, {}


def make_model(tensorboard_log: str, seed: int, n_steps: int, tb_log_name: str = "PPO") -> PPO:
    # n_steps here fixes the rollout buffer's capacity at construction
    # time. It MUST match the flush cadence train_dual() uses below
    # (also called n_steps there) — PPO's buffer requires exactly
    # `buffer.full` (i.e. buffer_size adds) before train() can call
    # rollout_buffer.get(), so passing a different value at either call
    # site desyncs the two and train() asserts.
    dummy_env = DummyVecEnv([lambda: _ShellEnv()])
    model = PPO(
        "MultiInputPolicy",
        dummy_env,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=seed,
    )
    # Bypassing model.learn() (see module docstring) means SB3 never runs
    # its own _setup_learn(), so model._logger is never created. Every
    # call below to model.train() reads self.logger internally (e.g. for
    # loss/clip-fraction records) and raises AttributeError on a fresh
    # model without this. configure_logger() is the same helper
    # _setup_learn() itself calls.
    model.set_logger(configure_logger(model.verbose, model.tensorboard_log, tb_log_name, reset_num_timesteps=True))
    return model


def _batch(obs: dict) -> dict:
    """Add the n_envs=1 axis SB3's buffers expect. Also transposes the
    image from (H, W, C) to (C, H, W) — normally SB3's VecTransposeImage
    wrapper does this automatically as observations flow through a VecEnv,
    but this loop bypasses that pipeline and calls the policy directly, so
    the transpose has to happen here instead."""
    out = {}
    for k, v in obs.items():
        if k == "image":
            v = np.transpose(v, (2, 0, 1))  # HWC -> CHW
        out[k] = v[None, ...]
    return out


def _predict_and_track(model: PPO, obs: dict):
    """One forward pass through the policy — returns the action to send to
    the game plus everything the rollout buffer needs to store."""
    obs_tensor = obs_as_tensor(_batch(obs), model.device)
    with torch.no_grad():
        actions, values, log_probs = model.policy(obs_tensor)
    action = actions.cpu().numpy()[0]
    return int(action), actions, values, log_probs


def _finish_rollout(model: PPO, last_obs: dict, done: bool, num_timesteps: int):
    """Bootstrap the value of the final state and run PPO's update. Mirrors
    OnPolicyAlgorithm.collect_rollouts()'s end-of-rollout bookkeeping."""
    obs_tensor = obs_as_tensor(_batch(last_obs), model.device)
    with torch.no_grad():
        _, last_values, _ = model.policy(obs_tensor)
    model.rollout_buffer.compute_returns_and_advantage(
        last_values=last_values, dones=np.array([done])
    )
    # learn() normally keeps num_timesteps current as rollouts are
    # collected; since this loop drives the buffer manually, do the same
    # here so the clip/LR schedules (which read _current_progress_remaining)
    # and the tensorboard x-axis are meaningful instead of frozen at 0.
    model.num_timesteps = num_timesteps
    model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
    model.train()
    # train() only writes into the logger's in-memory buffer — learn()
    # is normally what calls logger.dump() to flush it to tensorboard.
    # Without this, "python -m tensorboard" would show nothing for the
    # whole run even though training is proceeding correctly.
    model.logger.record("time/total_timesteps", model.num_timesteps)
    model.logger.dump(step=model.num_timesteps)
    model.rollout_buffer.reset()


def train_dual(
    total_ticks: int = 300_000,
    n_steps: int = 512,
    checkpoint_every: int = 20_000,
    checkpoint_dir: str = "./checkpoints_dual",
    tensorboard_log: str = "./logs_dual",
    mode: str = "training",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_p1 = make_model(tensorboard_log, seed=1, n_steps=n_steps, tb_log_name="PPO_P1")
    model_p2 = make_model(tensorboard_log, seed=2, n_steps=n_steps, tb_log_name="PPO_P2")
    # Needed by _update_current_progress_remaining() in _finish_rollout();
    # normally set by _setup_learn(), which this manual loop never calls.
    model_p1._total_timesteps = total_ticks
    model_p2._total_timesteps = total_ticks

    engine = DualKOFEngine(mode=mode)
    obs_p1, obs_p2 = engine.reset()

    episode_start = True
    tick = 0
    rollout_step = 0
    episode_reward_p1 = 0.0
    episode_reward_p2 = 0.0

    try:
        while tick < total_ticks:
            action_p1, actions_t1, values_p1, logp_p1 = _predict_and_track(model_p1, obs_p1)
            action_p2, actions_t2, values_p2, logp_p2 = _predict_and_track(model_p2, obs_p2)

            next_obs_p1, next_obs_p2, reward_p1, reward_p2, terminated, truncated, info = (
                engine.step(action_p1, action_p2)
            )
            done = terminated or truncated

            model_p1.rollout_buffer.add(
                _batch(obs_p1), actions_t1.cpu().numpy(),
                np.array([reward_p1]), np.array([episode_start]),
                values_p1, logp_p1,
            )
            model_p2.rollout_buffer.add(
                _batch(obs_p2), actions_t2.cpu().numpy(),
                np.array([reward_p2]), np.array([episode_start]),
                values_p2, logp_p2,
            )

            episode_reward_p1 += reward_p1
            episode_reward_p2 += reward_p2
            obs_p1, obs_p2 = next_obs_p1, next_obs_p2
            episode_start = False
            tick += 1
            rollout_step += 1

            if done:
                print(
                    f"[tick {tick}] episode ended | "
                    f"P1 reward={episode_reward_p1:.2f} P2 reward={episode_reward_p2:.2f} | "
                    f"P1 bars={info['p1_bars']} P2 bars={info['p2_bars']}"
                )
                obs_p1, obs_p2 = engine.reset()
                episode_start = True
                episode_reward_p1 = 0.0
                episode_reward_p2 = 0.0

            if rollout_step >= n_steps:
                _finish_rollout(model_p1, obs_p1, done, tick)
                _finish_rollout(model_p2, obs_p2, done, tick)
                rollout_step = 0
                print(f"[tick {tick}] PPO update done for both agents")

            if tick % checkpoint_every == 0:
                model_p1.save(os.path.join(checkpoint_dir, f"p1_{tick}"))
                model_p2.save(os.path.join(checkpoint_dir, f"p2_{tick}"))
                print(f"[tick {tick}] checkpoint saved")

    finally:
        engine.close()
        model_p1.save(os.path.join(checkpoint_dir, "p1_final"))
        model_p2.save(os.path.join(checkpoint_dir, "p2_final"))
        print("Saved final P1 and P2 models.")


if __name__ == "__main__":
    train_dual(mode="training")