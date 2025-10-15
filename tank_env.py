from enum import IntEnum
from typing import List, Optional
import numpy as np
import gymnasium as gym

class ActionState(IntEnum):
    SHOOT = 6

class TankState(IntEnum):
    X = 0
    Y = 1
    GUN_ANGLE = 2
    AMMO = 3
    CHARGE = 4
    IS_ALIVE = 5

class BulletState(IntEnum):
    X = 0
    Y = 1
    DX = 2
    DY = 3

class TankEnv(gym.Env):
    MAX_BULLET_COMPONENT_SPEED_PPS = 300
    MAX_AMMO = 3
    TANK_SPEED = 10
    TANK_SIZE_PIXELS = 20
    GUN_ROTATE_SPEED = 5
    TANK_SIZE_FROM_CENTER = TANK_SIZE_PIXELS // 2
    MAX_CHARGE_TIME_STEPS = 200
    CHARGE_LOSS_RATE = 10
    CHARGE_SPEED_FACTOR = 1 / 20
    BASE_BULLET_SPEED = 5

    KILL_REWARD = 200
    SHOOT_REWARD = 1
    def __init__(self, size=1000, players=2):
        super(TankEnv, self).__init__()

        self.size = size
        self.players = players
        self._agent_states = np.zeros((players, 6), dtype=np.int32)
        self._bullet_states = {idx: np.empty((0, 4), dtype=np.float64) for idx in range(players)}
        self._agents_killed = set()

        agent_space = gym.spaces.Box(
            np.array([0, 0, 0, 0, 0, 0]),
            np.array([size - 1, size - 1, 360, self.MAX_AMMO, self.MAX_CHARGE_TIME_STEPS, 1]),
            shape=(6,),
            dtype=np.int32
        )
        bullet_space = gym.spaces.Box(
            np.array([0, 0, -self.MAX_BULLET_COMPONENT_SPEED_PPS, -self.MAX_BULLET_COMPONENT_SPEED_PPS]),
            np.array([size - 1, size - 1, self.MAX_BULLET_COMPONENT_SPEED_PPS, self.MAX_BULLET_COMPONENT_SPEED_PPS]),
            shape=(4,),
            dtype=np.float64
        )
        self.observation_space = gym.spaces.Dict(
            {
                "agent": agent_space,
                "opponents": gym.spaces.Sequence(agent_space, stack=True),
                "bullets": gym.spaces.Sequence(bullet_space, stack=True)
            }
        )

        self.action_space = gym.spaces.MultiBinary(7)
        self._action_to_delta = {
            0: np.array([1, 0, 0]),  # right
            1: np.array([0, 1, 0]),  # up
            2: np.array([-1, 0, 0]), # left
            3: np.array([0, -1, 0]), # down
            4: np.array([0, 0, 1]),  # increase shot angle
            5: np.array([0, 0, -1]), # decrease shot angle
            6: np.array([0, 0, 0]),  # shoot
        }

    def _get_obs(self, agent_idx: int):
        this_agent = self._agent_states[agent_idx]
        other_agents = np.delete(self._agent_states, [agent_idx] + list(self._agents_killed), axis=0)

        bullet_states = np.empty(shape=(0,4), dtype=np.float64)
        for owner, bullets in self._bullet_states.items():
            bullet_states = np.vstack((bullet_states, *bullets))

        return {
            "agent": this_agent,
            "bullets": bullet_states,
            "opponents": other_agents,
        }

    def _get_info(self):
        return {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        locations = self.np_random.integers(0, self.size, size=(self.players,2), dtype=np.int32)
        angles = np.ones((self.players,1), dtype=np.int32) * 90
        ammo = np.ones((self.players,1), dtype=np.int32) * self.MAX_AMMO
        charge_time = np.zeros((self.players,1), dtype=np.int32)
        alive_flags = np.ones((self.players,1), dtype=np.int32)

        self._agent_states = np.hstack((locations, angles, ammo, charge_time, alive_flags))
        self._bullet_states = {idx: np.empty(shape=(0,4), dtype=np.float64) for idx in range(self.players)}
        self._agents_killed = set()

        return self._get_obs(0), self._get_info()

    def step(self, action):
        rewards = []

        agents_killed = set()
        for agent_idx in range(self.players):
            if self._agent_states[agent_idx][TankState.IS_ALIVE] == 0:
                continue

            reward, killed = self._step_agent(action, agent_idx)
            rewards.append(reward)
            agents_killed.update(killed)

        for agent_idx in agents_killed:
            self._agent_states[agent_idx][TankState.IS_ALIVE] = 0
            self._agents_killed.add(agent_idx)

        focused_agent = 0
        reward = rewards[focused_agent]
        truncated = False
        terminated = focused_agent in agents_killed
        observation = self._get_obs(focused_agent)
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _step_agent(self, action, agent_idx) -> tuple[int, List[int]]:
        reward = 0
        agents_to_kill = []

        bullets_to_destroy: List[int] = []
        for i, bullet in enumerate(self._bullet_states[agent_idx]):
            agent_killed = self._check_bullet_collision(bullet, agent_idx)
            if agent_killed != -1:
                reward += self.KILL_REWARD
                agents_to_kill.append(agent_killed)
                bullets_to_destroy.append(i)
            else:
                dx, dy = bullet[BulletState.DX], bullet[BulletState.DY]
                bullet[BulletState.X] += dx
                bullet[BulletState.Y] += dy

                if not (0 <= bullet[BulletState.X] < self.size) or not (0 <= bullet[BulletState.Y] < self.size):
                    bullets_to_destroy.append(i)

        if len(bullets_to_destroy) > 0:
            self._bullet_states[agent_idx] = np.delete(self._bullet_states[agent_idx], bullets_to_destroy, axis=0)

        agent_state = self._agent_states[agent_idx]
        did_move = False
        for action_idx, was_taken in enumerate(action):
            if was_taken == 1:
                direction = self._action_to_delta[action_idx]
                agent_state[TankState.X] += direction[TankState.X] * self.TANK_SPEED
                agent_state[TankState.Y] += direction[TankState.Y] * self.TANK_SPEED
                agent_state[TankState.GUN_ANGLE] += direction[TankState.GUN_ANGLE] * self.GUN_ROTATE_SPEED

                agent_state[TankState.X] = min(max(0, agent_state[TankState.X]), self.size - 1)
                agent_state[TankState.Y] = min(max(0, agent_state[TankState.Y]), self.size - 1)
                agent_state[TankState.GUN_ANGLE] = (agent_state[TankState.GUN_ANGLE] + 360) % 360

                did_move = did_move or direction[TankState.X] != 0 or direction[TankState.Y] != 0

        if not did_move:
            agent_state[TankState.CHARGE] += 1
            agent_state[TankState.CHARGE] = min(self.MAX_CHARGE_TIME_STEPS, agent_state[TankState.CHARGE])
        else:
            agent_state[TankState.CHARGE] -= self.CHARGE_LOSS_RATE
            agent_state[TankState.CHARGE] = max(0, agent_state[TankState.CHARGE])

        if action[ActionState.SHOOT] and agent_state[TankState.AMMO] > 0:
            speed = self.BASE_BULLET_SPEED + (agent_state[TankState.CHARGE] * self.CHARGE_SPEED_FACTOR)
            gun_angle_rad = np.deg2rad(agent_state[TankState.GUN_ANGLE])

            dx, dy = speed * np.cos(gun_angle_rad), speed * np.sin(gun_angle_rad)
            new_bullet = np.array([agent_state[TankState.X], agent_state[TankState.Y], dx, dy], dtype=np.float64)
            self._bullet_states[agent_idx] = np.vstack((self._bullet_states[agent_idx], new_bullet))

            agent_state[TankState.AMMO] -= 1
            reward += speed

        return reward, agents_to_kill

    def _check_bullet_collision(self, bullet, owner):
        bullet_x, bullet_y = bullet[BulletState.X], bullet[BulletState.Y]
        dx, dy = bullet[BulletState.DX], bullet[BulletState.DY]

        total_steps = max(abs(dx), abs(dy))
        x_step, y_step = dx / total_steps, dy / total_steps

        for _ in range(total_steps):
            for idx, player in enumerate(self._agent_states):
                if idx == owner:
                    continue

                if player[TankState.IS_ALIVE] == 0:
                    continue

                player_x, player_y = player[TankState.X], player[TankState.Y]
                if abs(bullet_x - player_x) <= self.TANK_SIZE_FROM_CENTER and abs(bullet_y - player_y) <= self.TANK_SIZE_FROM_CENTER:
                    return idx

            bullet_x += x_step
            bullet_y += y_step

        return -1

