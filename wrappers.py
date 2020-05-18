import gym
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
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
