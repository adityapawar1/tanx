from enum import IntEnum
from typing import Dict, List, Optional
import numpy as np
import gymnasium as gym
import pygame
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# TODO: refactor to not use agent_idx and just use agent_id
class TankEnv(MultiAgentEnv):
    RENDER_FPS = 20

    MAX_BULLET_COMPONENT_SPEED_PPS = 300
    MAX_AMMO = 3

    TANK_SPEED = 3
    TANK_SIZE_PIXELS = 30
    GUN_ROTATE_SPEED = 5
    TANK_SIZE_FROM_CENTER = TANK_SIZE_PIXELS // 2

    MAX_CHARGE_MULTIPLIER = 5
    MAX_CHARGE_TIME_STEPS = RENDER_FPS * 3
    CHARGE_LOSS_RATE = MAX_CHARGE_TIME_STEPS // RENDER_FPS
    CHARGE_SPEED_FACTOR = (MAX_CHARGE_MULTIPLIER - 1) // MAX_CHARGE_TIME_STEPS

    BASE_BULLET_SPEED = 20
    BULLET_RADIUS = 5
    GUN_SIZE_PIXELS = 20

    KILL_REWARD = 200
    SHOOT_REWARD = 1

    AGENT_PREFIX = "tank"

    metadata = {"render_modes": ["human"], "render_fps": RENDER_FPS}
    def __init__(self, config=None, render_mode=None, size=750, players=4):
        super(TankEnv, self).__init__()

        self.size = size
        self.players = players
        self.agents = self.possible_agents = [self.idx_to_agent_id(i) for i in range(players)]

        self._agent_states = np.zeros((players, 6), dtype=np.int32)
        self._bullet_states: Dict[int, np.ndarray] = {idx: np.empty((0, 4), dtype=np.float64) for idx in range(players)}
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
        self._action_to_delta: Dict[int, np.ndarray] = {
            ActionState.RIGHT: np.array([1, 0, 0]),
            ActionState.UP: np.array([0, 1, 0]),
            ActionState.LEFT: np.array([-1, 0, 0]),
            ActionState.DOWN: np.array([0, -1, 0]),
            ActionState.INCREASE_ANGLE: np.array([0, 0, 1]),
            ActionState.DECREASE_ANGLE: np.array([0, 0, -1]),
            ActionState.SHOOT: np.array([0, 0, 0]),
        }

        self.observation_spaces = {agent_id: self.observation_space for agent_id in self.possible_agents}
        self.action_spaces = {agent_id: self.action_space for agent_id in self.possible_agents}

        try:
            assert render_mode is None or render_mode in self.metadata["render_modes"]
        except:
            raise Exception(f"render mode is bad: {render_mode}")

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def agent_id_to_idx(self, agent_id: str) -> int:
        return int(agent_id[len(self.AGENT_PREFIX):])

    def idx_to_agent_id(self, agent_idx) -> str:
        return f"{self.AGENT_PREFIX}{agent_idx}"

    def _get_obs(self, agent_idx: int):
        this_agent = self._agent_states[agent_idx]
        other_agents = np.delete(self._agent_states, [agent_idx] + list(self._agents_killed), axis=0)

        bullet_states = np.empty(shape=(0,4), dtype=np.float64)
        for _, bullets in self._bullet_states.items():
            bullet_states = np.vstack((bullet_states, *bullets))

        return {
            "agent": this_agent,
            "bullets": bullet_states,
            "opponents": other_agents,
        }

    def _get_all_obs(self):
        return {agent_id: self._get_obs(self.agent_id_to_idx(agent_id)) for agent_id in self.agents}

    def _get_info(self):
        return {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        locations = self.np_random.integers(0, self.size, size=(self.players,2), dtype=np.int32)
        angles = np.ones((self.players,1), dtype=np.int32) * 90
        ammo = np.ones((self.players,1), dtype=np.int32) * self.MAX_AMMO
        charge_time = np.zeros((self.players,1), dtype=np.int32)
        alive_flags = np.ones((self.players,1), dtype=np.int32)

        self.agents = self.possible_agents

        self._agent_states = np.hstack((locations, angles, ammo, charge_time, alive_flags))
        self._bullet_states = {idx: np.empty(shape=(0,4), dtype=np.float64) for idx in range(self.players)}
        self._agents_killed = set()

        if self.render_mode == "human":
            self.render()

        return self._get_all_obs(), self._get_info()

    def step(self, action_dict):
        rewards = {}

        agents_killed = set()
        for agent_idx in range(self.players):
            action = action_dict.get(self.idx_to_agent_id(agent_idx), np.zeros(shape=(6,)))

            reward, killed = self._step_agent(action, agent_idx)
            rewards[self.idx_to_agent_id(agent_idx)] = reward
            agents_killed.update(killed)

        for agent_idx in agents_killed:
            self._agent_states[agent_idx][TankState.IS_ALIVE] = 0
            self.agents.remove(self.idx_to_agent_id(agent_idx))
            self._agents_killed.add(agent_idx)

        truncated = {}
        terminateds = {agent_id: self.agent_id_to_idx(agent_id) in agents_killed for agent_id in self.possible_agents}
        observations = self._get_all_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminateds, truncated, info

    def _step_agent(self, action, agent_idx) -> tuple[int, List[int]]:
        agent_state = self._agent_states[agent_idx]
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
                    agent_state[TankState.AMMO] += 1

        if len(bullets_to_destroy) > 0:
            self._bullet_states[agent_idx] = np.delete(self._bullet_states[agent_idx], bullets_to_destroy, axis=0)

        # still move bullets if they die, but don't move themselves
        if self._agent_states[agent_idx][TankState.IS_ALIVE] == 0:
            return 0, agents_to_kill

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
            speed = self.BASE_BULLET_SPEED * (1 + agent_state[TankState.CHARGE] * self.CHARGE_SPEED_FACTOR)
            gun_angle_rad = np.deg2rad(agent_state[TankState.GUN_ANGLE])

            dx, dy = speed * np.cos(gun_angle_rad), speed * np.sin(gun_angle_rad)
            dx = max(min(self.MAX_BULLET_COMPONENT_SPEED_PPS, dx), -self.MAX_BULLET_COMPONENT_SPEED_PPS)
            dy = max(min(self.MAX_BULLET_COMPONENT_SPEED_PPS, dy), -self.MAX_BULLET_COMPONENT_SPEED_PPS)

            new_bullet = np.array([agent_state[TankState.X], agent_state[TankState.Y], dx, dy], dtype=np.float64)
            self._bullet_states[agent_idx] = np.vstack((self._bullet_states[agent_idx], new_bullet))

            agent_state[TankState.AMMO] -= 1
            reward += speed

        return reward, agents_to_kill

    def _check_bullet_collision(self, bullet, owner):
        bullet_x, bullet_y = bullet[BulletState.X], bullet[BulletState.Y]
        dx, dy = bullet[BulletState.DX], bullet[BulletState.DY]

        # TODO: maybe dont just int() this
        total_steps = int(max(abs(dx), abs(dy)))
        x_step, y_step = dx / total_steps, dy / total_steps

        for _ in range(total_steps):
            for idx, player in enumerate(self._agent_states):
                if idx == owner:
                    continue

                if player[TankState.IS_ALIVE] == 0:
                    continue

                player_x, player_y = player[TankState.X], player[TankState.Y]
                collision_distance = self.TANK_SIZE_FROM_CENTER + self.BULLET_RADIUS
                if abs(bullet_x - player_x) <= collision_distance and abs(bullet_y - player_y) <= collision_distance:
                    return idx

            bullet_x += x_step
            bullet_y += y_step

        return -1

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.size, self.size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.size, self.size))
        canvas.fill((255, 255, 255))

        for agent_idx in range(self.players):
            agent_state = self._agent_states[agent_idx]
            if agent_state[TankState.IS_ALIVE] == 0:
                continue

            center_x, center_y = agent_state[TankState.X], agent_state[TankState.Y]
            left = center_x - self.TANK_SIZE_FROM_CENTER
            top = center_y - self.TANK_SIZE_FROM_CENTER

            gun_angle_rad = np.deg2rad(agent_state[TankState.GUN_ANGLE])
            gun_end_x = center_x + self.GUN_SIZE_PIXELS * np.cos(gun_angle_rad)
            gun_end_y = center_y + self.GUN_SIZE_PIXELS * np.sin(gun_angle_rad)

            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(left, top, self.TANK_SIZE_PIXELS, self.TANK_SIZE_PIXELS),
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (center_x, center_y),
                (gun_end_x, gun_end_y),
                3
            )

        for _, bullets in self._bullet_states.items():
            for bullet in bullets:
                center_x, center_y = bullet[BulletState.X], bullet[BulletState.Y]

                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (center_x, center_y),
                    self.BULLET_RADIUS
                )

        if self.window is not None and self.clock is not None:
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class ActionState(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    INCREASE_ANGLE = 4
    DECREASE_ANGLE = 5
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

