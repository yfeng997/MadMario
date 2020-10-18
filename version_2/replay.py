import os
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
import datetime

from metrics import MetricLogger
from wrappers import ResizeObservation, SkipFrame
from agent import Mario

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

env.reset()

save_dir = os.path.join(
    "checkpoints",
    f"{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

"Load Pre-trained Mario Net"
# possible loading path
# checkpoints/2020-10-13T00-53-30
# checkpoints/2020-10-15T00-12-19
# checkpoints/2020-10-17T01-44-25
load_path = "checkpoints/2020-10-15T00-12-19/mario_net_0.chkpt" # original from checkpoints/2020-10-13T00-53-30
mario.load(load_path)
mario.exploration_rate = 0.138

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state = env.reset()

    while True:

        env.render()

        action = mario.act(state)

        next_state, reward, done, info = env.step(action)

        mario.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None)

        state = next_state

        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
