from typing import Any, Literal
from enum import Enum
import numpy as np
import gymnasium as gym
import pygame

class SnakeEnv(gym.Env):
    MAX_TIME_SEC = 120
    RENDER_FPS = 7
    GRID_SIZE = 40

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}
    def __init__(self, board_size=12, starting_length=4, num_apples=3, render_mode=None, absolute_direction=False):
        self.size = board_size
        self.starting_length = starting_length
        self.num_apples = num_apples
        self.absolute_direction = absolute_direction

        self._direction: Direction = Direction.UP
        self._body_locations = np.array([[-1, -1]], dtype=np.int32)
        self._apple_locations = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Box(low=-self.size, high=-self.size, shape=(self.num_apples*2+1,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(3)
        self._current_step = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        angle = self._direction.angle_rad()

        apple_relative_pos = self._apple_locations - self._body_locations[0]
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        apple_relative = np.rint(np.matmul(rotation_matrix, apple_relative_pos.T).T)

        return np.append(apple_relative, len(self._body_locations)).astype(np.int32)

    def _get_info(self):
        return {}

    def _wrapped_add(self, pos, delta):
        x = pos[0] + delta[0]
        y = pos[1] + delta[1]

        return np.array([x % self.size, y % self.size], dtype=np.int32)

    def _create_apple(self, n=1, initial=False):
        if initial:
            self._apple_locations = np.empty((2,), dtype=np.int32)

        locations = []
        for _ in range(n):
            apple_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)

            # TODO: generate list of all valid locations and np.choose
            while self._is_in(apple_location, self._body_locations) and self._is_in(apple_location, self._apple_locations):
                apple_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)
            locations.append(apple_location)

        return np.array(locations)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[object, dict[str, Any]]:
        super().reset(seed=seed)

        start_direction = Direction(self.np_random.integers(0, 4))
        delta = start_direction.delta()
        start_head_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)
        start_locations = []
        for i in range(self.starting_length):
            start_locations.append(self._wrapped_add(start_head_location, delta * -i))

        self._body_locations = np.array(start_locations)
        self._apple_locations = self._create_apple(n=self.num_apples, initial=True)
        self._direction = start_direction
        self._current_step = 0
        self.window = None
        self.clock = None

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self._current_step += 1

        print(f"current step = {self._current_step}")

        if action != Direction.NONE.value:
            if self.absolute_direction:
                self._direction = Direction(action)
            else:
                self._direction = self._direction.relative_to(action)

        new_head_location = self._wrapped_add(self._body_locations[0], self._direction.delta())

        died = self._is_in(new_head_location, self._body_locations)

        new_body_locations = np.insert(self._body_locations, 0, new_head_location, axis=0)
        ate_apple = self._is_in(new_head_location, self._apple_locations)
        if not ate_apple:
            new_body_locations = np.delete(new_body_locations, -1, axis=0)
        self._body_locations = new_body_locations

        if ate_apple:
            mask = [not np.array_equal(apple, new_head_location) for apple in self._apple_locations]
            new_apple_locations = self._apple_locations[mask]
            new_apple_locations = np.insert(new_apple_locations, 0, self._create_apple(), axis=0)
            self._apple_locations = new_apple_locations

        observation = self._get_obs()
        apple_reward = ((len(self._body_locations) - self.starting_length) / self.size ** 2)
        reward = apple_reward if ate_apple else 0
        terminated = died or len(self._body_locations) == self.size ** 2
        truncated = self._current_step > self.MAX_TIME_SEC * self.RENDER_FPS
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _pos_to_px(self, pos):
        return pos[0] * self.GRID_SIZE, pos[1] * self.GRID_SIZE

    def _pos_to_px_center(self, pos):
        return (pos[0] * self.GRID_SIZE) + self.GRID_SIZE // 2, (pos[1] * self.GRID_SIZE) + self.GRID_SIZE // 2

    def _is_in(self, target, arr):
        for item in arr:
            if np.array_equal(target, item):
                return True
        return False

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.size * self.GRID_SIZE, self.size * self.GRID_SIZE)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.size * self.GRID_SIZE, self.size * self.GRID_SIZE))
        canvas.fill((255, 255, 255))

        for i in range(self.size):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, i * self.GRID_SIZE),
                (self.size * self.GRID_SIZE, i * self.GRID_SIZE),
                1
            )

        for i in range(self.size):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (i * self.GRID_SIZE, 0),
                (i * self.GRID_SIZE, self.size * self.GRID_SIZE),
                1
            )

        for i, body in enumerate(self._body_locations):
            x, y = self._pos_to_px(body)
            pygame.draw.rect(canvas, (0, 111, 230),
                        pygame.Rect(x, y, self.GRID_SIZE, self.GRID_SIZE))

            if i == 0:
                eye_x, eye_y = self._pos_to_px_center(body)
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (eye_x, eye_y),
                    5
                )

        for apple in self._apple_locations:
            x, y = self._pos_to_px_center(apple)
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (x, y),
                5
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
            self.window = None
            self.clock = None


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NONE = 4

    def relative_to(self, action: Literal[0, 1, 2]):
        dir = Direction(action)
        if action == 2:
            dir = Direction.LEFT

        if dir == Direction.UP:
            return self
        elif dir == Direction.RIGHT:
            return Direction((self.value + 1) % 4)
        else:
            return Direction((self.value - 1) % 4)

    def angle_rad(self):
        if self == Direction.UP:
            return np.deg2rad(0)
        elif self == Direction.RIGHT:
            return np.deg2rad(90)
        elif self == Direction.DOWN:
            return np.deg2rad(-90)
        else:
            return np.deg2rad(180)

    def delta(self):
        if self == Direction.UP:
            return np.array([0, -1], dtype=np.int32)
        elif self == Direction.RIGHT:
            return np.array([1, 0], dtype=np.int32)
        elif self == Direction.DOWN:
            return np.array([0, 1], dtype=np.int32)
        else:
            return np.array([-1, 0], dtype=np.int32)


