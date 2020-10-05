# Gym is an OpenAI toolkit for RL
import gym
# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
import torch
from torch import nn
import random, datetime, numpy as np, cv2

from gym.wrappers import FrameStack, GrayScaleObservation

#NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

from metrics import clear_metrics, collect_metrics, display_metrics, log_metrics
from agent import Mario
from wrappers import ResizeObservation, SkipFrame

# Initialize Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

# Apply Wrappers to environment
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)
env = SkipFrame(env, skip=4)

env.reset()

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda} \n")

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n)

episodes = 10000

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = env.reset()
    metrics = clear_metrics()

    # Play the game!
    while True:

        # 3. Show environment (the visual) [WIP]
        #env.render()

        # 4. Run agent on the state
        action = mario.act(state)

        # 5. Agent performs action
        next_state, reward, done, info = env.step(action)

        # 6. Remember
        mario.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = mario.learn()

        # 8. Loggings
        metrics = collect_metrics(metrics, reward, loss, q)

        # 9. Update state
        state = next_state

        # If done break loop
        if done or info['flag_get']:
            print(
                f"{display_metrics(metrics)} | "
                f"Total Experiences: {mario.nb_steps} | "
                f"{datetime.datetime.now().strftime('%H:%M:%S')} "
            )

            log_metrics(metrics, mario.save_dir)

            break
