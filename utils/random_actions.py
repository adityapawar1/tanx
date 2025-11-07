import gymnasium as gym
from gymnasium.utils.env_checker import check_env

gym.register(
    id="SnakeEnv-v0",
    entry_point="envs.snake_env:SnakeEnv",
    max_episode_steps=10_000,
)

if __name__ == '__main__':
    env = gym.make("SnakeEnv-v0", render_mode="human")
    check_env(env.unwrapped)

    observation, info = env.reset()
    episode_over = False

    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        env.render()

    env.close()

