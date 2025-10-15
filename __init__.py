import gymnasium as gym

gym.register(
    id="TankEnv-v0",
    entry_point="tank_env:TankEnv",
    # max_episode_steps=10_000,
)
