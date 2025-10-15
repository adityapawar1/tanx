from ray.tune.registry import register_env
from tank_env import TankEnv

register_env("TankEnv-v0", lambda config: TankEnv(config))

