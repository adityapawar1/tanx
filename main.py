import gymnasium as gym
from gymnasium.utils.env_checker import check_env

gym.register(
    id="TankEnv-v0",
    entry_point="tank_env:TankEnv",
    # max_episode_steps=10_000,
)

if __name__ == '__main__':
    env = gym.make("TankEnv-v0", render_mode="human")
    # check_env(env.unwrapped)

    observation, info = env.reset()
    episode_over = False
    total_reward: int = 0

    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(action, reward)

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()

