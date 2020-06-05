
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from wrappers import wrapper
from agent import DQNAgent

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env,
    # [['NOOP'],
    [['right'],
    ['right', 'A']]
    # ['right', 'B'],]
)
env = wrapper(env)

state_dim = (4, 84, 84)
action_dim = env.action_space.n

agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, max_memory=100000, double_q=True)

for agent_idx in range(agent.save_total):
    agent.replay(env, load_idx=agent_idx)
