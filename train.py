import os
import multiprocessing
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune import RunConfig, CheckpointConfig
from ray.rllib.algorithms.appo import APPOConfig
from tank_env import TankEnv
import dotenv

dotenv.load_dotenv()

if __name__ == "__main__":
    s3_bucket_name = os.environ["S3_BUCKET_NAME"]
    register_env("TankEnv-v0", lambda config: TankEnv(config))
    env = TankEnv()

    num_cpus = multiprocessing.cpu_count()
    ray.init(address="auto")

    resources = ray.cluster_resources()
    total_cpus = int(resources.get("CPU", 1))
    print(f"Detected {total_cpus} total CPUs across cluster.")
    num_env_runners = max(1, total_cpus - 2)

    config = (
        APPOConfig()
        .training(
            lr=1e-7,
            train_batch_size_per_learner=24000,
            num_epochs=1,
            vtrace=True,
            use_kl_loss=False,

            lambda_=0.98,
            clip_param=0.2,
            grad_clip=40.0,
            grad_clip_by="global_norm",

            vf_loss_coeff=0.5,
            entropy_coeff=0.001,
        )
        .environment("TankEnv-v0")
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=8,
            rollout_fragment_length="auto"
        )
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
            },
            policy_mapping_fn=lambda agent_id, *a, **k: "standard_policy",
        )
    )

    config.observation_filter = "MeanStdFilter"

    run_config = RunConfig(
        name="tank-target-regen-v10",
        storage_path=f"s3://{s3_bucket_name}/ray",
        stop={"training_iteration": 5000},
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

    tuner.fit()

