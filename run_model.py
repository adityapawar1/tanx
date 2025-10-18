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
    # local_path = "/Users/adityapawar_1/Documents/work/personal/tank/ray_results/tank-target-objective-v6/PPO_TankEnv-v0_1fe7b_00000_0_2025-10-16_22-50-02/checkpoint_000001"
    # local_path = "./ray_results/tank-relative-obs-v7/PPO_TankEnv-v0_1cc5e_00000_0_2025-10-17_03-36-17/checkpoint_000027"
    local_path = "./ray_results/tank-hyperparams-v9/PPO_TankEnv-v0_0ce0a_00000_0_2025-10-17_16-00-18/checkpoint_000001"
    local_path = "./ray_results/tank-hyperparams-v9/PPO_TankEnv-v0_0ce0a_00000_0_2025-10-17_16-00-18/checkpoint_000002"
    local_path = "./ray_results/tank-hyperparams-v9/PPO_TankEnv-v0_c17d3_00000_0_2025-10-17_18-57-09/checkpoint_000014"
    local_path = "/Users/adityapawar_1/Documents/work/personal/tank/ray_results/tank-hyperparams-v9/PPO_TankEnv-v0_5cd93_00000_0_2025-10-17_21-31-49/checkpoint_000010"
    local_path = "/Users/adityapawar_1/Documents/work/personal/tank/ray_results/tank-hyperparams-v9/PPO_TankEnv-v0_5cd93_00000_0_2025-10-17_21-31-49/checkpoint_000147"

    checkpoint_path = os.path.abspath(local_path)
    test_trained_model(checkpoint_path, num_episodes=3)



