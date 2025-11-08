import gymnasium as gym
import numpy as np
import pygame
from envs.snake_env import Direction

gym.register(
    id="SnakeEnv-v0",
    entry_point="envs.snake_env:SnakeEnv",
    max_episode_steps=10_000,
)

def direction_from_pressed(pressed):
    if pressed[pygame.K_w]:
        return Direction.UP
    elif pressed[pygame.K_r]:
        return Direction.DOWN
    elif pressed[pygame.K_a]:
        return Direction.LEFT
    elif pressed[pygame.K_s]:
        return Direction.RIGHT
    else:
        return Direction.NONE

def run(env):
    obs, info = env.reset()

    while True:
        pressed = pygame.key.get_pressed()
        action = direction_from_pressed(pressed)
        print(action)
        obs, reward, terminated, truncated, info = env.step(action.value)
        print(f"{obs=}")
        print(f"{reward=}")

        env.render()
        if terminated or truncated:
            break

    env.close()

if __name__ == '__main__':
    env = gym.make("SnakeEnv-v0", render_mode="human", absolute_direction=True)
    run(env)



