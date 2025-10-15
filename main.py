import gymnasium as gym
from gymnasium.utils.env_checker import check_env

gym.register(
    id="TankEnv-v0",
    entry_point="tank_env:TankEnv",
    max_episode_steps=10_000,
)

if __name__ == '__main__':
    env = gym.make("TankEnv-v0")
    check_env(env.unwrapped)
