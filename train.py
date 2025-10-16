import multiprocessing
from typing import Optional
import ray
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

    num_cpus = multiprocessing.cpu_count()
    ray.init(
        num_cpus=num_cpus,
        local_mode=False,
        ignore_reinit_error=True
    )

    config = (
        PPOConfig()
        .environment("TankEnv-v0")
        .framework("torch")
        .multi_agent(
            policies={
                "standard_policy":
                    (
                        None,
                        env.observation_space,
                        env.action_space,
                        {
                        "model": {
                            "vf_share_layers": True,
                            "use_lstm": False,
                            "free_log_std": False,
                            "fcnet_hiddens": [256, 256],
                            "fcnet_activation": "relu",
                        }
                    }
                ),
                "lstm_policy":
                    (
                        None,
                        env.observation_space,
                        env.action_space,
                        {
                        "model": {
                            "vf_share_layers": True,
                            "use_lstm": True,
                            "free_log_std": False,
                            "fcnet_hiddens": [256, 256],
                            "fcnet_activation": "relu",
                        }
                    }
                )

            },
            policy_mapping_fn=lambda agent_id, *a, **k: "lstm_policy" if agent_id in ["tank0", "tank1"] else "shared_policy",
        )
        .env_runners(
            num_env_runners=num_cpus - 1,
            num_envs_per_env_runner=2
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_num_env_runners=2,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
        )
        .resources()
    )

    run_config = RunConfig(
        name="tank-better-reward-v4",
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

