import os
import numpy as np
import pygame
from ray.tune.registry import register_env
from ray.rllib.core import (
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
)
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from tank_env import TankEnv

def play_trained_model(checkpoint_path):
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

    user_agent = env.possible_agents[0]
    while True:
        obs, info = env.reset()

        while len(env.agents) > 1:
            actions = {}
            for agent_id in env.agents:
                action = []
                if agent_id == user_agent:
                    pressed = pygame.key.get_pressed()
                    # colemak wasd :(
                    action = np.array([
                        pressed[pygame.K_s],
                        pressed[pygame.K_r],
                        pressed[pygame.K_a],
                        pressed[pygame.K_w],
                        pressed[pygame.K_e],
                        pressed[pygame.K_n],
                        pressed[pygame.K_i],
                    ])
                else:
                    rl_module = policies[agent_id]
                    fwd_outputs= rl_module.forward_inference({'obs': torch.Tensor(obs[agent_id]).unsqueeze(0)})
                    action_dist_class = rl_module.get_inference_action_dist_cls()
                    action_dist = action_dist_class.from_logits(
                        fwd_outputs["action_dist_inputs"]
                    )
                    action = action_dist.sample()[0].numpy()
                    action = np.zeros((7,))

                actions[agent_id] = action

            obs, rewards, terminated, truncated, info = env.step(actions)


    env.close()

if __name__ == '__main__':
    register_env("TankEnv-v0", lambda config: TankEnv(config))
    # local_path = "./ray_results/tank-relative-obs-v7/PPO_TankEnv-v0_1cc5e_00000_0_2025-10-17_03-36-17/checkpoint_000027"
    local_path = "./ray_results/tank-hyperparams-v9/PPO_TankEnv-v0_c17d3_00000_0_2025-10-17_18-57-09/checkpoint_000014"

    checkpoint_path = os.path.abspath(local_path)
    play_trained_model(checkpoint_path)



