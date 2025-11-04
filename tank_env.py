from enum import IntEnum
from typing import Dict, List, Optional
import numpy as np
import gymnasium as gym
import pygame
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# TODO: refactor to not use agent_idx and just use agent_id
# TODO: make ts bounce
class TankEnv(MultiAgentEnv):
    RENDER_FPS = 40
    MAX_TIME_SECONDS = 100

    MAX_HEALTH = 3
    TANK_SPEED = 2
    TANK_SIZE_PIXELS = 32
    TANK_SIZE_FROM_CENTER = TANK_SIZE_PIXELS // 2

    MAX_CHARGE_MULTIPLIER = 4
    MAX_CHARGE_TIME_STEPS = RENDER_FPS * 2
    CHARGE_LOSS_RATE = (MAX_CHARGE_TIME_STEPS * 2) // RENDER_FPS
    CHARGE_SPEED_FACTOR = (MAX_CHARGE_MULTIPLIER - 1) / MAX_CHARGE_TIME_STEPS

    BULLET_RADIUS = 5
    BASE_BULLET_SPEED = 10
    GUN_SIZE_PIXELS = 20
    GUN_ROTATE_SPEED = 6
    SHOOT_COOLDOWN = int(0.2 * RENDER_FPS)
    MAX_BULLET_COMPONENT_SPEED_PPS = 300
    MAX_AMMO = 3
    AMMO_REPLENISH_TIME_STEPS = RENDER_FPS * 1.5

    TARGET_WIDTH = 100
    TARGET_SPEED_FACTOR = 1.4
    TARGET_REGEN_SECONDS = 4

    WIN_REWARD = 0.5
    KILL_REWARD = 1.0
    DEATH_PENALTY = -1.0
    HURT_PENALTY = -0.2
    HIT_REWARD = 0.8
    TARGET_REGEN_REWARD = 0.7
    SURVIVAL_REWARD = 0.00 / RENDER_FPS
    TARGET_REWARD = 0.02 / RENDER_FPS
    TARGET_DISTANCE_REWARD_MULTIPLIER = 1.5 / RENDER_FPS
    MOVE_REWARD = 0.000 / RENDER_FPS
    SHOOT_PENALTY = -0.0
    BULLET_SPEED_REWARD_FACTOR = 0.02

    AGENT_PREFIX = "tank"
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}
    def __init__(self, config=None, render_mode=None, size=600, players=4):
        super(TankEnv, self).__init__()

        self.size = size
        self.players = players
        self.possible_agents = [self.idx_to_agent_id(i) for i in range(players)]

        self._current_step = 0
        self._agent_states = np.zeros((self.players, 6), dtype=np.int32)
        self._target_state = np.zeros((2,), dtype=np.int32)
        self._shoot_cooldown_states = np.zeros((self.players,), dtype=np.int32)
        self._bullet_states: Dict[int, np.ndarray] = {idx: np.empty((0, 4), dtype=np.float32) for idx in range(players)}
        self._agents_killed = set()
        self._target_streaks = np.zeros((self.players,), dtype=np.int32)
        self._ammo_replenish_counters = np.zeros((self.players,), dtype=np.int32)
        self._agent_info = [{"agent_kills": 0, "max_charge": 0, "agent_hits": 0} for _ in range(self.players)]

        targeted_agent_space_low = np.array([0, 0, 0, 0, 0, 0])
        opponent_agent_space_low = np.array([-(size - 1), -(size - 1), 0, 0, 0, 0])
        agent_space_high = np.array([size - 1, size - 1, 360, self.MAX_AMMO, self.MAX_CHARGE_TIME_STEPS, self.MAX_HEALTH])

        single_bullet_space_low = np.array([-(size - 1), -(size - 1), -self.MAX_BULLET_COMPONENT_SPEED_PPS, -self.MAX_BULLET_COMPONENT_SPEED_PPS])
        single_bullet_space_high = np.array([size - 1, size - 1, self.MAX_BULLET_COMPONENT_SPEED_PPS, self.MAX_BULLET_COMPONENT_SPEED_PPS])

        self.target_space_low = np.array([-(size - 1), -(size - 1)])
        self.target_space_high = np.array([size + 1, size + 1])
        self.full_agent_space_low = np.hstack((targeted_agent_space_low, np.tile(opponent_agent_space_low, players - 1)))
        self.full_agent_space_high = np.tile(agent_space_high, players)
        self.full_bullet_space_low = np.tile(single_bullet_space_low, (players - 1) * self.MAX_AMMO)
        self.full_bullet_space_high = np.tile(single_bullet_space_high, (players - 1) * self.MAX_AMMO)

        self.obs_space_low = np.concat((self.target_space_low, self.full_agent_space_low, self.full_bullet_space_low))
        self.obs_space_high = np.concat((self.target_space_high, self.full_agent_space_high, self.full_bullet_space_high))

        self.observation_space = gym.spaces.Box(
            low=self.obs_space_low / self.obs_space_high,
            high=self.obs_space_high / self.obs_space_high,
            shape=(len(self.obs_space_high),),
            dtype=np.float32
        )

        # ray rllib doesnt support multi binary rn bruh
        # self.action_space = gym.spaces.MultiBinary(7)
        self.action_space = gym.spaces.MultiDiscrete([2] * 7)
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

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def agent_id_to_idx(self, agent_id: str) -> int:
        return int(agent_id[len(self.AGENT_PREFIX):])

    def idx_to_agent_id(self, agent_idx) -> str:
        return f"{self.AGENT_PREFIX}{agent_idx}"

    def _get_info(self, agent_idx):
        return self._agent_info[agent_idx]

    def _get_obs(self, agent_idx: int):
        this_agent = self._agent_states[agent_idx]
        this_agent_pos = np.array([this_agent[TankState.X], this_agent[TankState.Y]])
        this_agent_angle_deg = this_agent[TankState.GUN_ANGLE]
        this_agent_angle_rad = np.deg2rad(this_agent[TankState.GUN_ANGLE])

        rotation_matrix = np.array([
            [np.cos(-this_agent_angle_rad), -np.sin(-this_agent_angle_rad)],
            [np.sin(-this_agent_angle_rad), np.cos(-this_agent_angle_rad)]
        ])

        other_agents = np.delete(self._agent_states, agent_idx, axis=0)
        other_agents_relative = []
        for other_agent in other_agents:
            pos = np.array([other_agent[TankState.X], other_agent[TankState.Y]])
            angle_deg = other_agent[TankState.GUN_ANGLE]
            relative_pos = pos - this_agent_pos
            angled_pos = np.matmul(rotation_matrix, relative_pos.T).T
            relative_angle = (angle_deg - this_agent_angle_deg + 360) % 360
            relative_agent_state = np.hstack((angled_pos, relative_angle, other_agent[TankState.GUN_ANGLE+1:]))

            other_agents_relative.extend(relative_agent_state)
        other_agents_relative = np.array(other_agents_relative)

        target_relative = self._target_state - this_agent_pos
        target_relative = np.matmul(rotation_matrix, target_relative.T).T

        bullet_states_relative = []
        for owner, bullets in self._bullet_states.items():
            if agent_idx == owner:
                continue

            for bullet in bullets:
                pos = np.array([bullet[BulletState.X], bullet[BulletState.Y]])
                velocity = np.array([bullet[BulletState.DX], bullet[BulletState.DY]])
                relative_pos = pos - this_agent_pos
                angled_pos = np.matmul(rotation_matrix, relative_pos.T).T
                angled_speed = np.matmul(rotation_matrix, velocity.T).T
                relative_state = np.hstack((angled_pos, angled_speed))

                bullet_states_relative.extend(relative_state)

        bullet_states_relative = np.pad(bullet_states_relative, (0, len(self.full_bullet_space_low) - len(bullet_states_relative)))

        # if agent_idx == 0:
        #     print(f"{target_relative=}")
        #     print(f"{this_agent=}")
        #     for i in range(self.players - 1):
        #         state = other_agents_relative[i*6:(i+1)*6]
        #         print(f"agent {i}: pos=({state[0]:.1f}, {state[1]:.1f}) angle={state[2]:.1f} ammo={state[3]} charge={state[4]} alive={state[5]}")
        #
        #     for i in range((len(bullet_states_relative)) // 4):
        #         state = bullet_states_relative[i*4:(i+1)*4]
        #         print(f"bullet {i}: pos=({state[0]:.1f}, {state[1]:.1f}) speed=({state[2]:.1f}, {state[3]:.1f})")

        return np.concat((
            target_relative,
            this_agent,
            other_agents_relative,
            bullet_states_relative,
        )) / self.obs_space_high

    def _get_all_info(self):
        return {agent_id: self._get_info(self.agent_id_to_idx(agent_id)) for agent_id in self.agents}

    def _get_all_obs(self):
        return {agent_id: self._get_obs(self.agent_id_to_idx(agent_id)) for agent_id in self.agents}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        locations = self.np_random.integers(0, self.size, size=(self.players,2), dtype=np.int32)
        target = self.np_random.integers(int(self.size * 0.1), int(self.size * 0.9), size=(2,), dtype=np.int32)
        angles = self.np_random.integers(0, 360, size=(self.players,1), dtype=np.int32)
        ammo = np.ones((self.players,1), dtype=np.int32) * self.MAX_AMMO
        charge_time = np.zeros((self.players,1), dtype=np.int32)
        health = np.ones((self.players,1), dtype=np.int32) * self.MAX_HEALTH

        self.agents = self.possible_agents[:]

        self._current_step = 0
        self._agent_states = np.hstack((locations, angles, ammo, charge_time, health))
        self._shoot_cooldown_states = np.zeros((self.players,), dtype=np.int32)
        self._target_streaks = np.zeros((self.players,), dtype=np.int32)
        self._target_state = target
        self._bullet_states = {idx: np.empty(shape=(0,4), dtype=np.float32) for idx in range(self.players)}
        self._agents_killed = set()
        self._ammo_replenish_counters = np.zeros((self.players,), dtype=np.int32)
        self._agent_info = [{"agent_kills": 0, "max_charge": 0, "agent_hits": 0} for _ in range(self.players)]

        if self.render_mode == "human":
            self.render()

        return self._get_all_obs(), self._get_all_info()

    def step(self, action_dict):
        self._current_step += 1
        rewards = {}

        agents_hit = set()
        for agent_idx in range(self.players):
            action = np.array(action_dict.get(self.idx_to_agent_id(agent_idx), [0] * 7))

            reward, killed = self._step_agent(action, agent_idx)
            if agent_idx not in self._agents_killed:
                rewards[self.idx_to_agent_id(agent_idx)] = reward
            agents_hit.update(killed)

        observations = self._get_all_obs()
        info = self._get_all_info()

        for agent_idx in agents_hit:
            self._agent_states[agent_idx][TankState.HEALTH] -= 1
            rewards[self.idx_to_agent_id(agent_idx)] += self.HURT_PENALTY

            if self._agent_states[agent_idx][TankState.HEALTH] == 0:
                self.agents.remove(self.idx_to_agent_id(agent_idx))
                self._agents_killed.add(agent_idx)
                rewards[self.idx_to_agent_id(agent_idx)] += self.DEATH_PENALTY

        time_cutoff = self._current_step > self.RENDER_FPS * self.MAX_TIME_SECONDS
        truncated = {agent_id: time_cutoff for agent_id in self.agents}
        terminateds = {agent_id: self.agent_id_to_idx(agent_id) in self._agents_killed for agent_id in self.possible_agents}
        terminateds["__all__"] = len(self.agents) <= 1
        if terminateds["__all__"] and len(self.agents) == 1:
            rewards[self.agents[0]] += self.WIN_REWARD

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminateds, truncated, info

    def _calculate_target_reward(self, agent_state) -> tuple[bool, float]:
        agent_x, agent_y = agent_state[TankState.X], agent_state[TankState.Y]
        target_x, target_y = self._target_state[0], self._target_state[1]

        threshold = self.TARGET_WIDTH // 2 + self.TANK_SIZE_FROM_CENTER
        is_on_target = bool(abs(agent_x - target_x) <= threshold and abs(agent_y - target_y) <= threshold)
        distance_to_target = max(1, np.hypot(agent_x - target_x, agent_y - target_y) * 0.5)

        return is_on_target, (1/distance_to_target) * self.TARGET_DISTANCE_REWARD_MULTIPLIER

    def _step_agent(self, action, agent_idx) -> tuple[float, List[int]]:
        agent_state = self._agent_states[agent_idx]
        reward = self.SURVIVAL_REWARD
        agents_hit = set()

        bullets_to_destroy: List[int] = []
        for i, bullet in enumerate(self._bullet_states[agent_idx]):
            agent_hit = self._check_bullet_collision(bullet, agent_idx)
            if agent_hit != -1:
                reward += self.HIT_REWARD
                agents_hit.add(agent_hit)
                bullets_to_destroy.append(i)
                if self._agent_states[agent_hit][TankState.HEALTH] - 1 == 0:
                    reward += self.KILL_REWARD
                    self._agent_info[agent_idx]["agent_kills"] += 1

                self._agent_info[agent_idx]["agent_hits"] += 1
            else:
                dx, dy = bullet[BulletState.DX], bullet[BulletState.DY]
                bullet[BulletState.X] += dx
                bullet[BulletState.Y] += dy

                if not (0 <= bullet[BulletState.X] < self.size) or not (0 <= bullet[BulletState.Y] < self.size):
                    bullets_to_destroy.append(i)

        if len(bullets_to_destroy) > 0:
            self._bullet_states[agent_idx] = np.delete(self._bullet_states[agent_idx], bullets_to_destroy, axis=0)

        # still move bullets if they die, but don't move themselves
        if self._agent_states[agent_idx][TankState.HEALTH] == 0:
            return 0, list(agents_hit)

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

        if not did_move and agent_state[TankState.AMMO] > 0:
            agent_state[TankState.CHARGE] += 1
            agent_state[TankState.CHARGE] = min(self.MAX_CHARGE_TIME_STEPS, agent_state[TankState.CHARGE])
            self._agent_info[agent_idx]["max_charge"] = max(self._agent_info[agent_idx]["max_charge"], agent_state[TankState.CHARGE])
        else:
            reward += self.MOVE_REWARD
            agent_state[TankState.CHARGE] -= self.CHARGE_LOSS_RATE
            agent_state[TankState.CHARGE] = max(0, agent_state[TankState.CHARGE])

        if self._shoot_cooldown_states[agent_idx] >= 1:
            self._shoot_cooldown_states[agent_idx] -= 1

        is_on_target, target_distance_reward = self._calculate_target_reward(agent_state)
        reward += target_distance_reward
        if is_on_target:
            reward += self.TARGET_REWARD
            if agent_state[TankState.HEALTH] != self.MAX_HEALTH:
                self._target_streaks[agent_idx] += 1
        else:
            self._target_streaks[agent_idx] = 0

        if self._target_streaks[agent_idx] >= self.RENDER_FPS * self.TARGET_REGEN_SECONDS:
            reward += self.TARGET_REGEN_REWARD
            agent_state[TankState.HEALTH] = min(self.MAX_HEALTH, agent_state[TankState.HEALTH] + 1)
            self._target_streaks[agent_idx] = 0

        if action[ActionState.SHOOT] and agent_state[TankState.AMMO] > 0 and self._shoot_cooldown_states[agent_idx] == 0:
            charge_multiplier = 1 + agent_state[TankState.CHARGE] * self.CHARGE_SPEED_FACTOR
            if is_on_target:
                charge_multiplier *= self.TARGET_SPEED_FACTOR

            speed = self.BASE_BULLET_SPEED * charge_multiplier
            gun_angle_rad = np.deg2rad(agent_state[TankState.GUN_ANGLE])

            dx, dy = speed * np.cos(gun_angle_rad), speed * np.sin(gun_angle_rad)
            dx = max(min(self.MAX_BULLET_COMPONENT_SPEED_PPS, dx), -self.MAX_BULLET_COMPONENT_SPEED_PPS)
            dy = max(min(self.MAX_BULLET_COMPONENT_SPEED_PPS, dy), -self.MAX_BULLET_COMPONENT_SPEED_PPS)

            new_bullet = np.array([agent_state[TankState.X], agent_state[TankState.Y], dx, dy], dtype=np.float32)
            self._bullet_states[agent_idx] = np.vstack((self._bullet_states[agent_idx], new_bullet))

            agent_state[TankState.AMMO] -= 1

            reward += self.SHOOT_PENALTY
            reward += ((charge_multiplier - 1) * self.BULLET_SPEED_REWARD_FACTOR)

            agent_state[TankState.CHARGE] = 0
            self._shoot_cooldown_states[agent_idx] = self.SHOOT_COOLDOWN

        if agent_state[TankState.AMMO] < self.MAX_AMMO:
            self._ammo_replenish_counters[agent_idx] += 1
            if self._ammo_replenish_counters[agent_idx] >= self.AMMO_REPLENISH_TIME_STEPS:
                agent_state[TankState.AMMO] += 1
                self._ammo_replenish_counters[agent_idx] = 0

        return reward, list(agents_hit)


    def _check_bullet_collision(self, bullet, owner):
        bullet_x, bullet_y = bullet[BulletState.X], bullet[BulletState.Y]
        dx, dy = bullet[BulletState.DX], bullet[BulletState.DY]

        # TODO: maybe dont just int() this
        total_steps = max(1, int(np.hypot(dx, dy)))
        x_step, y_step = dx / total_steps, dy / total_steps

        for _ in range(total_steps):
            for idx, player in enumerate(self._agent_states):
                if idx == owner:
                    continue

                if player[TankState.HEALTH] == 0:
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

        target_left = self._target_state[0] - self.TARGET_WIDTH // 2
        target_top = self._target_state[1] - self.TARGET_WIDTH // 2
        pygame.draw.rect(canvas, (0, 111, 230),
                        pygame.Rect(target_left, target_top, self.TARGET_WIDTH, self.TARGET_WIDTH))

        for agent_idx in range(self.players):
            agent_state = self._agent_states[agent_idx]
            if agent_state[TankState.HEALTH] == 0:
                continue

            center_x, center_y = agent_state[TankState.X], agent_state[TankState.Y]
            tank_left = center_x - self.TANK_SIZE_FROM_CENTER
            tank_top = center_y - self.TANK_SIZE_FROM_CENTER

            gun_angle_rad = np.deg2rad(agent_state[TankState.GUN_ANGLE])
            gun_end_x = center_x + self.GUN_SIZE_PIXELS * np.cos(gun_angle_rad)
            gun_end_y = center_y + self.GUN_SIZE_PIXELS * np.sin(gun_angle_rad)

            pygame.draw.rect(
                canvas,
                (255, 20, 0),
                pygame.Rect(tank_left, tank_top, self.TANK_SIZE_PIXELS, self.TANK_SIZE_PIXELS),
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (center_x, center_y),
                (gun_end_x, gun_end_y),
                3
            )

            ammo = agent_state[TankState.AMMO]
            ammo_x = center_x - self.TANK_SIZE_PIXELS
            for i in range(ammo):
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (ammo_x, i * 15 + center_y - self.TANK_SIZE_FROM_CENTER),
                    4
                )

            charge = agent_state[TankState.CHARGE]
            charge_y = center_y - self.TANK_SIZE_FROM_CENTER - 10
            charge_length = self.TANK_SIZE_PIXELS * (charge / self.MAX_CHARGE_TIME_STEPS)

            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (tank_left, charge_y),
                (tank_left + self.TANK_SIZE_PIXELS, charge_y),
                5
            )

            if charge_length >= 2:
                pygame.draw.line(
                    canvas,
                    (252, 152, 3),
                    (tank_left, charge_y),
                    (tank_left + charge_length, charge_y),
                    5
                )

            target_streak = self._target_streaks[agent_idx]
            if target_streak > 0:
                target_streak_y = center_y - self.TANK_SIZE_FROM_CENTER - 20
                target_streak_length = (self.TANK_SIZE_PIXELS + 16) * (target_streak / (self.RENDER_FPS * self.TARGET_REGEN_SECONDS))
                target_streak_left = center_x - self.TANK_SIZE_FROM_CENTER - 8

                pygame.draw.line(
                    canvas,
                    (0, 0, 0),
                    (target_streak_left, target_streak_y),
                    (target_streak_left + self.TANK_SIZE_PIXELS + 16, target_streak_y),
                    5
                )

                if target_streak_length >= 2:
                    pygame.draw.line(
                        canvas,
                        (252, 152, 40),
                        (target_streak_left, target_streak_y),
                        (target_streak_left + target_streak_length, target_streak_y),
                        5
                    )

            health = agent_state[TankState.HEALTH]
            health_y = center_y + self.TANK_SIZE_FROM_CENTER + 10
            health_length = self.TANK_SIZE_PIXELS * (health / self.MAX_HEALTH)

            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (tank_left, health_y),
                (tank_left + self.TANK_SIZE_PIXELS, health_y),
                5
            )
            pygame.draw.line(
                canvas,
                (252, 0, 0),
                (tank_left, health_y),
                (tank_left + health_length, health_y),
                5
            )

        for _, bullets in self._bullet_states.items():
            for bullet in bullets:
                center_x, center_y = int(bullet[BulletState.X]), int(bullet[BulletState.Y])

                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (center_x, center_y),
                    self.BULLET_RADIUS
                )

        if self.render_mode == "human" and self.window is not None and self.clock is not None:
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

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
    HEALTH = 5

class BulletState(IntEnum):
    X = 0
    Y = 1
    DX = 2
    DY = 3

