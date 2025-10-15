import os
from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.envs.env_rendering_and_recording import EnvRenderCallback

from tank_env import TankEnv

if __name__ == '__main__':
    env = TankEnv()

    config = (
        PPOConfig()
        .environment(TankEnv)
        .multi_agent(
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies={"shared_policy": (None, env.observation_space, env.action_space, {})}
        )
        .callbacks(EnvRenderCallback)
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            lr=3e-4,
        )
    )
    config.evaluation(
        evaluation_interval=1,
        evaluation_num_env_runners=2,
        evaluation_duration_unit="episodes",
        evaluation_duration=10,
    )

    if config.is_multi_agent:
        print("Training multi agent environment")
    else:
        print("Training single agent environment")

    ppo = config.build_algo()

    for i in range(3):
        pprint(ppo.train())

        checkpoint_path = os.path.abspath(f"checkpoints/episode{i}")
        checkpoint_dir = ppo.save(checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_dir}")
