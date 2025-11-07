from types import NoneType
from typing import Any, Literal, Optional
from enum import Enum
import numpy as np
import gymnasium as gym
import pygame

class SnakeEnv(gym.Env):
    RENDER_FPS = 5
    GRID_SIZE = 40

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}
    def __init__(self, board_size=20, starting_length=4, num_apples=1, render_mode=None):
        self.size = board_size
        self.starting_length = starting_length
        self.num_apples = num_apples

        self._body_locations = np.array([[-1, -1]], dtype=np.int32)
        self._direction: Direction = Direction.UP
        self._apple_locations = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Box(low=0, high=20, shape=(11,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(3)
        self._current_step = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return np.zeros(shape=(11,), dtype=np.int32)

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
            while apple_location not in self._body_locations and apple_location not in self._apple_locations:
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
            start_locations.append(self._wrapped_add(start_head_location, delta * i))

        self._body_locations = np.array(start_locations)
        self._apple_locations = self._create_apple(n=self.num_apples, initial=True)
        self._direction = start_direction
        self._current_step = 0
        self.window = None
        self.clock = None

        return self._get_obs(), self._get_info()

    def step(self, action):
        self._current_step += 1

        print(f"current step = {self._current_step}")
        self._direction = self._direction.relative_to(action)
        print(f"current dir = {self._direction}")
        new_head_location = self._wrapped_add(self._body_locations[0], self._direction.delta())

        print(f"body = {self._body_locations}")
        print(f"found apple = {new_head_location in self._apple_locations}")
        died = new_head_location in self._body_locations
        new_body_locations = np.insert(self._body_locations, 0, new_head_location, axis=0)
        if new_head_location not in self._apple_locations:
            new_body_locations = np.delete(new_body_locations, -1, axis=0)

        self._body_locations = new_body_locations

        observation = self._get_obs()
        reward = 0
        terminated = died
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _pos_to_px(self, pos):
        return pos[0] * self.GRID_SIZE, pos[1] * self.GRID_SIZE

    def _pos_to_px_center(self, pos):
        return (pos[0] * self.GRID_SIZE) + self.GRID_SIZE // 2, (pos[1] * self.GRID_SIZE) + self.GRID_SIZE // 2

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

        for body in self._body_locations:
            x, y = self._pos_to_px(body)
            pygame.draw.rect(canvas, (0, 111, 230),
                        pygame.Rect(x, y, self.GRID_SIZE, self.GRID_SIZE))

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

    def delta(self):
        if self == Direction.UP:
            return np.array([0, -1], dtype=np.int32)
        elif self == Direction.RIGHT:
            return np.array([1, 0], dtype=np.int32)
        elif self == Direction.DOWN:
            return np.array([0, 1], dtype=np.int32)
        else:
            return np.array([-1, 0], dtype=np.int32)


