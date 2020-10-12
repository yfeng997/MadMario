import os
import gym_super_mario_bros

from gym.wrappers import FrameStack, GrayScaleObservation

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
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)
env = SkipFrame(env, skip=4)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=None)

load_dir = "2020-10-08T11-00-32"

for mario_idx in range(mario.save_total):
    load_path = os.path.join(load_dir, f"mario_net_{mario_idx}.chkpt")
    mario.load(load_path)
    mario.exploration_rate = mario.exploration_rate_min
    mario.curr_step = 0

    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        action = mario.act(state=state)
        next_state, reward, done, info = env.step(action=action)
        total_reward += reward
        state = next_state
        if done or info['flag_get']:
            break
    print(
        f"#{mario_idx}. Mario finished after {mario.curr_step} steps "
        f"with a total reward of {total_reward}"
    )
