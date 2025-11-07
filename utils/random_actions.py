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
    agent_ids = ["tank0", "tank1", "tank2", "tank3"]


    while not episode_over:
        action_dict = {agent_id: env.action_space.sample() for agent_id in agent_ids}
        env.step(action_dict)

    env.close()

