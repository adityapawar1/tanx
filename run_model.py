from ray.rllib.algorithms.ppo import PPO
from tank_env import TankEnv
from ray.tune import ExperimentAnalysis

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
    analysis = ExperimentAnalysis("./ray_results/tank")
    best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max")
    if best_trial:
        checkpoint_path = analysis.get_best_checkpoint(
            trial=best_trial,
            metric="episode_reward_mean", mode="max"
        )

        test_trained_model(checkpoint_path, num_episodes=3)
