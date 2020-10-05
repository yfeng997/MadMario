import gym
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation
from gym.spaces import Box
import cv2

import pdb

class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


class ResizeObservation(gym.ObservationWrapper):
    """Downsample the image observation to a square image. """
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation


def wrapper(env):
    # skip to every 4th frame. Remove redundant info. to speed up training
    env = SkipEnv(env, skip=4)
    # rgb to gray. Reduce input dimension thus model size
    env = GrayScaleObservation(env, keep_dim=False)
    # resize to 84 x 84. Reduce input dimension thus model size
    env = ResizeObservation(env, shape=84)
    # make obs a stack of previous 3 frames. Need consecutive frames
    # to differentiate landing vs. taking off
    env = FrameStack(env, num_stack=4)
    return env
