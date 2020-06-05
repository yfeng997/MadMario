from collections import deque
import torch
import torch.nn as nn
import numpy as np
import random
from neural import ConvNet
import os
import datetime

import pdb

class DQNAgent:
    def __init__(self, state_dim, action_dim, max_memory, double_q):
        # state space dimension
        self.state_dim = state_dim
        # action space dimension
        self.action_dim = action_dim
        # replay buffer
        self.memory = deque(maxlen=max_memory)
        # if double_q, use best action from online_q for next state q value
        self.double_q = double_q
        # future reward discount rate
        self.gamma = 0.9
        # initial epsilon(random exploration rate)
        self.eps = 1
        # final epsilon
        self.eps_min = 0.1
        # epsilon decay rate
        self.eps_decay = 0.99999975
        # current step, updated everytime the agent acts
        self.step = 0
        # number of experiences between updating online q
        self.learn_every = 3
        # number of experiences to collect before training
        self.burnin = 1e5
        # self.burnin = 1e2
        # number of experiences between updating target q with online q
        self.sync_every = 1e4
        # number of experiences between saving the current agent
        self.save_every = 1e5
        # number of consecutive marios to save
        self.save_total = 5
        # a new directory to save marios to
        self.save_dir = os.path.join(
            "checkpoints",
            f"{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
        )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # batch size used to update online q
        self.batch_size = 32
        # online action value function, Q(s, a)
        self.online_q = ConvNet(input_dim=state_dim, output_dim=action_dim)
        # target action value function, Q'(s, a)
        self.target_q = ConvNet(input_dim=state_dim, output_dim=action_dim)
        # optimizer
        self.optimizer = torch.optim.Adam(self.online_q.parameters(), lr=0.00025)

    def predict(self, state, model):
        """Given a state, predict Q values of all actions
        model is either 'online' or 'target'
        """
        state_float = torch.tensor(np.array(state)).float() / 255.
        if model == 'online':
            return self.online_q(state_float)
        if model == 'target':
            return self.target_q(state_float)

    def act(self, state):
        """Given a state, choose an epsilon-greedy action
        """
        if np.random.rand() < self.eps:
            # random action
            action = np.random.randint(low=0, high=self.action_dim)
        else:
            # policy action
            q = self.predict(np.expand_dims(state, 0), model='online')
            action = torch.max(q, axis=1)[1].item()
        # decrease eps
        self.eps *= self.eps_decay
        self.eps = max(self.eps_min, self.eps)
        # increment step
        self.step += 1
        return action

    def remember(self, experience):
        """Add the observation to memory
        """
        self.memory.append(experience)

    def learn(self):
        """Update online action value (Q) function with a batch of experiences
        """
        # sync target network
        if self.step % self.sync_every == 0:
            self.sync_target_q()
        # checkpoint model
        if self.step % self.save_every < self.save_total:
            self.save_model()
        # break if burn-in
        if self.step < self.burnin:
            return None, None
        # break if no training
        if self.step % self.learn_every != 0:
            return None, None
        # sample batch
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))
        # get next q values from target_q
        next_q = self.predict(next_state, 'target')
        # calculate discounted future reward
        if self.double_q:
            q = self.predict(next_state, 'online')
            q_idx = torch.max(q, axis=1)[1]
            target_q = torch.tensor(reward) + torch.tensor(1. - done) * self.gamma * next_q[np.arange(0, self.batch_size), q_idx]
        else:
            target_q = torch.tensor(reward) + torch.tensor(1. - done) * self.gamma * torch.max(next_q, axis=1)[0]
        # get predicted q values from online_q and actions taken
        curr_q = self.predict(state, 'online')
        pred_q = curr_q[np.arange(0, self.batch_size), action]
        # huber loss
        loss = nn.functional.smooth_l1_loss(pred_q, target_q)
        # update online_q
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return pred_q.mean().item(), loss.item()


    def save_model(self):
        """Save the current agent
        """
        save_path = os.path.join(self.save_dir, f"online_q_{self.step % self.save_total}.chkpt")
        torch.save(self.online_q.state_dict(), save_path)


    def sync_target_q(self):
        """Update target action value (Q) function with online action value (Q) function
        """
        self.target_q.load_state_dict(self.online_q.state_dict())


    def replay(self, env, load_dir=None, load_idx=0):
        if not load_dir:
            dirs = os.listdir("checkpoints")
            load_dir = sorted(dirs)[-1]
        load_path = os.path.join("checkpoints", load_dir, f"online_q_{load_idx}.chkpt")

        if not os.path.exists(load_path):
            return
        state_dict = torch.load(load_path)
        self.online_q.load_state_dict(state_dict)
        self.eps = self.eps_min

        state = env.reset()
        total_reward = 0
        while True:
            env.render()
            action = self.act(state=state)
            next_state, reward, done, info = env.step(action=action)
            total_reward += reward
            state = next_state
            if done or info['flag_get']:
                break
        return total_reward
