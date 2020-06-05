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
        mean_reward = np.round(np.mean(log['rewards'][-100:]), 5)
        mean_length = np.round(np.mean(log['lengths'][-100:]), 5)
        mean_loss = np.round(np.mean(log['losses'][-100:]), 5)
        mean_q_value = np.round(np.mean(log['q_values'][-100:]), 5)
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

        log_file = os.path.join(agent.save_dir, "log.json")
        if not os.path.exists(log_file):
            json_log = {
                "rewards": [mean_reward],
                "lengths": [mean_length],
                "losses": [mean_loss],
                "q_values": [mean_q_value]
            }
        else:
            with open(log_file) as f:
                json_log = json.load(f)
                json_log["rewards"].append(mean_reward)
                json_log["lengths"].append(mean_length)
                json_log["losses"].append(mean_loss)
                json_log["q_values"].append(mean_q_value)

        with open(log_file, 'w') as f:
            json.dump(json_log, f, indent=2)
