from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from wrappers import wrapper
from agent import DQNAgent
import numpy as np
import pdb
import time
import cv2
import os
import json


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
# Action Space:
# NOOP: no action
# right: walk right
# right, A: jump right
# right, B: run right
# Observation Space: 240 x 256 x 3
env = JoypadSpace(env,
    [['right'],
    ['right', 'A']]
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

# Log
log = {
    "rewards": [],
    "lengths": [],
    "losses": [],
    "q_values": []
}
log_file = os.path.join(agent.save_dir, "log.txt")
with open(log_file, "w") as f:
    f.write(f"{'MeanReward':>15}{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}\n")

# Timing
start = time.time()
step = 0

# Main loop
for e in range(episodes):

    # Reset env
    state = env.reset()

    # Logging
    ep_reward = 0.0
    ep_length = 0
    ep_total_loss = 0.0
    ep_total_q = 0.0
    ep_learn_length = 1 # used for mean loss/q_value

    # Play
    while True:

        # Show env
        # env.render()

        # Run agent
        action = agent.act(state=state)

        # Perform action
        next_state, reward, done, info = env.step(action=action)

        # Remember
        agent.remember(experience=(state, next_state, action, reward, done))

        # Learn (conditional)
        q_value, loss = agent.learn()

        # Logging
        ep_reward += reward
        ep_length += 1
        if q_value and loss:
            ep_total_loss += loss
            ep_total_q += q_value
            ep_learn_length += 1

        # Update state
        state = next_state

        # If done break loop
        if done or info['flag_get']:
            break

    # Log
    log["rewards"].append(ep_reward)
    log["lengths"].append(ep_length)
    log["losses"].append(np.round(ep_total_loss/ep_learn_length, 5))
    log["q_values"].append(np.round(ep_total_q/ep_learn_length, 5))

    # Print & Log
    if e % 50 == 0:
        mean_reward = np.round(np.mean(log['rewards'][-100:]), 3)
        mean_length = np.round(np.mean(log['lengths'][-100:]), 3)
        mean_loss = np.round(np.mean(log['losses'][-100:]), 3)
        mean_q_value = np.round(np.mean(log['q_values'][-100:]), 3)
        print(
            f"Episode {e} - "
            f"Step {agent.step} - "
            f"Step/sec {np.round((agent.step - step) / (time.time() - start))} - "
            f"Epsilon {np.round(agent.eps, 3)} - "
            f"Mean Reward {mean_reward} - "
            f"Mean Length {mean_length} - "
            f"Mean Loss {mean_loss} - "
            f"Mean Q Value {mean_q_value}"
        )
        start = time.time()
        step = agent.step

        with open(log_file, "a") as f:
            f.write(f"{mean_reward:15.3f}{mean_length:15.3f}{mean_loss:15.3f}{mean_q_value:15.3f}\n")
