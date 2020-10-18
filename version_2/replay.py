import os
import gym_super_mario_bros

from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from nes_py.wrappers import JoypadSpace
from wrappers import ResizeObservation, SkipFrame

from agent import Mario

"""Environment definition is copied entirely from main.py
"""
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=None)

load_dir = "2020-10-17T01-44-25"

episodes = 100
total_reward = 0.0
total_length = 0.0
for _ in range(episodes):
    load_path = os.path.join(load_dir, f"mario_net_1.chkpt")
    mario.load(load_path)
    mario.exploration_rate = 0.1
    mario.curr_step = 0

    state = env.reset()
    ep_reward = 0.0
    ep_length = 0.0
    while True:
        # env.render()
        action = mario.act(state=state)
        next_state, reward, done, info = env.step(action=action)
        ep_reward += reward
        ep_length += 1
        state = next_state
        if done or info['flag_get']:
            break
    total_reward += ep_reward
    total_length += ep_length

print(
    f"Replay finished with avg. length {total_length/episodes}, avg. reward {total_reward/episodes}"
)
