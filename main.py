from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from wrappers import wrapper
from agent import DQNAgent
import numpy as np
import pdb
import time


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
# action space: [0 (no action), 1 (walk right), 2 (jump right), 3 (run right)]
# observation space: 240 x 256 x 3
env = JoypadSpace(env,
    [['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],]
)
# observation space: 4 (#frame) x 84 (height) x 84 (width)
env = wrapper(env)

# Parameters
state_dim = (4, 84, 84)
action_dim = env.action_space.n

# Agent
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, max_memory=100000, double_q=True)

# Episodes
episodes = 10000
rewards = []

# Timing
start = time.time()
step = 0

# Main loop
for e in range(episodes):

    # Reset env
    state = env.reset()

    # Reward
    total_reward = 0
    iter = 0

    # Play
    while True:

        # Show env
        env.render()

        # Run agent
        action = agent.act(state=state)

        # Perform action
        next_state, reward, done, info = env.step(action=action)

        # Remember
        agent.remember(experience=(state, next_state, action, reward, done))

        # Replay
        agent.learn()

        # Total reward
        total_reward += reward

        # Update state
        state = next_state

        # Increment
        iter += 1

        # If done break loop
        if done or info['flag_get']:
            break

    # Rewards
    rewards.append(total_reward / iter)

    # Print
    if e % 100 == 0:
        print('Episode {e} - '
              'Frame {f} - '
              'Frames/sec {fs} - '
              'Epsilon {eps} - '
              'Mean Reward {r}'.format(e=e,
                                       f=agent.step,
                                       fs=np.round((agent.step - step) / (time.time() - start)),
                                       eps=np.round(agent.eps, 4),
                                       r=np.mean(rewards[-100:])))
        start = time.time()
        step = agent.step

# Save rewards
np.save('rewards.npy', rewards)
