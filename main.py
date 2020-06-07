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
import datetime
import sys


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
    f.write(
        f"{'Episode':>8}{'Step':>10}{'Epsilon':>10}{'MeanReward':>15}"
        f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}{'Time':>20}\n"
    )
start_time = time.time()
start_step = agent.step

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

        # Perform action (20% time cost)
        next_state, reward, done, info = env.step(action=action)

        # Remember
        agent.remember(experience=(state, next_state, action, reward, done))

        # Learn (conditional) (80% time cost)
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
        eps = np.round(agent.eps, 3)
        step_time = np.round((time.time() - start_time)/(agent.step - start_step), 3)
        start_time = time.time()
        start_step = agent.step
        print(
            f"Episode {e} - "
            f"Step {agent.step} - "
            f"Step Time {step_time} - "
            f"Epsilon {eps} - "
            f"Mean Reward {mean_reward} - "
            f"Mean Length {mean_length} - "
            f"Mean Loss {mean_loss} - "
            f"Mean Q Value {mean_q_value} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(log_file, "a") as f:
            f.write(
                f"{e:8d}{agent.step:10d}{eps:10.3f}"
                f"{mean_reward:15.3f}{mean_length:15.3f}{mean_loss:15.3f}{mean_q_value:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        # Running on Colab, download checkpoints to local
        if 'google.colab' in sys.modules:
            from google.colab import files
            files.download(os.path.join(agent.save_dir, "online_q_1.chkpt"))
