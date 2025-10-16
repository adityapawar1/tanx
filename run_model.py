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
    # local_path = "./ray_results/tank-better-reward-v4/PPO_TankEnv-v0_a9856_00000_0_2025-10-15_23-52-20/checkpoint_000004"
    # local_path = "./ray_results/tank-better-reward-v4/PPO_TankEnv-v0_a9856_00000_0_2025-10-15_23-52-20/checkpoint_000007"
    # local_path = "./ray_results/tank-better-reward-v4/PPO_TankEnv-v0_f7fe7_00000_0_2025-10-16_00-51-48/checkpoint_000019"
    local_path = "./ray_results/tank-scaled-reward-v5/PPO_TankEnv-v0_20890_00000_0_2025-10-16_11-37-10/checkpoint_000009"

    checkpoint_path = os.path.abspath(local_path)
    test_trained_model(checkpoint_path, num_episodes=3)



