from ray.rllib.algorithms.ppo import PPO
from tank_env import TankEnv

def test_trained_model(checkpoint_path, num_episodes=5):
    ppo = PPO.from_checkpoint(checkpoint_path)

    env = TankEnv(render_mode="human")

    for episode in range(num_episodes):
        obs, info = env.reset()

        while len(env.agents) > 1:
            actions = {}
            for agent_id in env.agents:
                action = ppo.compute_single_action(obs[agent_id], policy_id="shared_policy")
                actions[agent_id] = action

            obs, rewards, terminated, truncated, info = env.step(actions)


    env.close()

if __name__ == '__main__':
    test_trained_model("checkpoints/episode2", num_episodes=3)
