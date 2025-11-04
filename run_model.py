import os
from ray.tune.registry import register_env
from ray.rllib.core import (
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
)
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from tank_env import TankEnv

def test_trained_model(checkpoint_path, num_episodes=5):
    env = TankEnv(render_mode="human")

    policies = {agent_id: RLModule.from_checkpoint(
        os.path.join(
            checkpoint_path,
            COMPONENT_LEARNER_GROUP,
            COMPONENT_LEARNER,
            COMPONENT_RL_MODULE,
            "standard_policy",
        )
    ) for agent_id in env.possible_agents}

    for episode in range(num_episodes):
        obs, info = env.reset()

        while len(env.agents) > 1:
            actions = {}
            for agent_id in env.agents:
                rl_module = policies[agent_id]
                fwd_outputs= rl_module.forward_inference({'obs': torch.Tensor(obs[agent_id]).unsqueeze(0)})
                action_dist_class = rl_module.get_inference_action_dist_cls()
                action_dist = action_dist_class.from_logits(
                    fwd_outputs["action_dist_inputs"]
                )
                action = action_dist.sample()[0].numpy()
                actions[agent_id] = action

            obs, rewards, terminated, truncated, info = env.step(actions)


    env.close()

if __name__ == '__main__':
    register_env("TankEnv-v0", lambda config: TankEnv(config))
    # path = "./ray_results/tank-hyperparams-v9/PPO_TankEnv-v0_0ce0a_00000_0_2025-10-17_16-00-18/checkpoint_000001"
    # path = "./ray_results/tank-hyperparams-v9/PPO_TankEnv-v0_0ce0a_00000_0_2025-10-17_16-00-18/checkpoint_000002"
    # path = "./ray_results/tank-hyperparams-v9/PPO_TankEnv-v0_c17d3_00000_0_2025-10-17_18-57-09/checkpoint_000014"
    # path = "/Users/adityapawar_1/Documents/work/personal/tank/ray_results/tank-hyperparams-v9/PPO_TankEnv-v0_5cd93_00000_0_2025-10-17_21-31-49/checkpoint_000010"
    # path = "/Users/adityapawar_1/Documents/work/personal/tank/ray_results/tank-hyperparams-v9/PPO_TankEnv-v0_5cd93_00000_0_2025-10-17_21-31-49/checkpoint_000147"
    # path = "/Users/adityapawar_1/Documents/work/personal/tank/ray_results/tank-target-regen-v10/PPO_TankEnv-v0_99675_00000_0_2025-10-18_14-58-37/checkpoint_000001"
    # path = "/Users/adityapawar_1/Documents/work/personal/tank/ray_results/tank-target-regen-v10/PPO_TankEnv-v0_f3630_00000_0_2025-10-18_15-22-36/checkpoint_000007"
    # path = "/Users/adityapawar_1/Documents/work/personal/tank/ray_results/tank-target-regen-v10/PPO_TankEnv-v0_f3630_00000_0_2025-10-18_15-22-36/checkpoint_000011"
    path = "/Users/adityapawar_1/Documents/work/personal/tank/ray_results/tank-target-regen-v10/PPO_TankEnv-v0_f3630_00000_0_2025-10-18_15-22-36/checkpoint_000020"
    path = "/Users/adityapawar_1/Documents/work/personal/tank/ray_results/tank-kl-loss-v10.1/PPO_TankEnv-v0_ff09e_00000_0_2025-10-18_19-40-38/checkpoint_000149"
    path = "/Users/adityapawar_1/Documents/work/personal/tank/ray_results/tank-kl-loss-v10.2/PPO_TankEnv-v0_cd137_00000_0_2025-10-19_14-15-55/checkpoint_000008"

    checkpoint_path = os.path.abspath(path)
    test_trained_model(checkpoint_path, num_episodes=3)



