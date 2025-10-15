from typing import Optional
from ray import tune
from ray.tune.registry import register_env
from ray.tune import RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig
from tank_env import TankEnv
from ray.rllib.examples.envs.env_rendering_and_recording import EnvRenderCallback
import os

if __name__ == "__main__":
    register_env("TankEnv-v0", lambda config: TankEnv(config))
    env = TankEnv()

    config = (
        PPOConfig()
        .environment("TankEnv-v0")
        .framework("torch")
        .training(
            train_batch_size=4000,
            # sgd_minibatch_size=128,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
        )
        .multi_agent(
            policies={"shared_policy": (None, env.observation_space, env.action_space, {})},
            policy_mapping_fn=lambda agent_id, *a, **k: "shared_policy",
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_num_env_runners=2,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
        )
        # .callbacks(EnvRenderCallback)
    )

    run_config = RunConfig(
        name="tank-v0",
        storage_path=os.path.abspath("./ray_results"),
        stop={"training_iteration": 1000},
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=25,
            checkpoint_at_end=True,
            num_to_keep=3,
        ),
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=run_config,
    )

    results_dir = os.path.abspath("./ray_results/tank-v0")
    # if os.path.exists(results_dir):
    #     tuner = tune.Tuner.restore(results_dir, resume_errored=True, trainable=)
    # else:

    tuner.fit()

