from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig

from tank_env import TankEnv

if __name__ == '__main__':
    config = (
        PPOConfig()
        .environment(TankEnv)
        .multi_agent()
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
        ppo.save_to_path(f"checkpoints/episode{i}")
