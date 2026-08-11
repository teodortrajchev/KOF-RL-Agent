from kof_env import KOFEnv
#smeneto od env.kof_env
import time

env = KOFEnv(render_mode="human", mode="training")

obs, info = env.reset()
print("RESET OK: image", obs["image"].shape, "| vector", obs["vector"].shape)

total_reward = 0

for i in range(50):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    print(
        f"step={i:03d} "
        f"action={action} "
        f"reward={reward:.3f} "
        f"hp_p={info['player_hp']:.2f} "
        f"hp_e={info['enemy_hp']:.2f} "
        f"pos_valid={info['position_valid']} "
        f"enemy_right={info['enemy_is_right']}"
    )

    env.render()

    if terminated or truncated:
        print("Episode ended early")
        break

env.close()

print("TOTAL REWARD:", total_reward)