import os
from ray import tune
from ray.tune.registry import register_env
from tank_env import TankEnv

if __name__ == '__main__':
    register_env("TankEnv-v0", lambda config: TankEnv(config))
    env = TankEnv()

    config = {
        "env": "TankEnv-v0",
        "framework": "torch",
        "num_env_runners": 2,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "multiagent": {
            "policies": {
                "shared_policy": (None, env.observation_space, env.action_space, {})
            },
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: "shared_policy",
        },
        "evaluation_interval": 10,
        "evaluation_num_env_runners": 2,
        "evaluation_duration": 10,
        "evaluation_duration_unit": "episodes",
    }

    tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 1000},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        storage_path=os.path.abspath("./ray_results"),
        name="tank",
    )
