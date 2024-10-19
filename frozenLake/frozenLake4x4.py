import random

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

# Create environment,
# 如果is_slippery=True，那么冰面是滑的，那么做出行为后有1/3的可能性滑倒其他地方
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array")
# Record video every 100 episode
env = RecordVideo(env, 'videos/', episode_trigger=lambda x: x % 500 == 0)

# 这个场景中有四个行为，就是 上/下/左/右
env.reset()
env.render()
# build up Q-table
# 地图是4*4的这意味着有16种状态，乘上4个动作，所以Q-table是16*4的
# 创建一个Q-table，初始化为0
qtable = np.zeros((16, 4))
# qtable = np.random.rand(16, 4)

# !!! Alternatively, the gym library can also directly g
# give us the number of states and actions using
# "env.observation_space.n" and "env.action_space.n"
nb_states = env.observation_space.n  # = 16
nb_actions = env.action_space.n  # = 4
qtable2 = np.zeros((nb_states, nb_actions))

# Let's see how it looks
print('Q-table =')
print(qtable)

# 0: Move left
# 1: Move down
# 2: Move right
# 3: Move up
action = env.action_space.sample()

# 2. Implement this action and move the agent in the desired direction
observation, reward, terminated, truncated, info = env.step(action)

# Display the results (reward and map)
env.render()
print(f'Reward = {reward}')

# 导入matplotlib用来分析
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100
plt.rcParams.update({'font.size': 10})

# Hyperparameters
episodes = 2001 # Total number of episodes
alpha = 0.001  # Learning rate α
gamma = 0.9  # Discount factor γ
epsilon = 1.0  # Exploration rate ε
epsilon_decay = 0.0001  # Decay rate of ε

# List of outcomes to plot
outcomes = []
for _ in tqdm(range(episodes)):
    state = env.reset()
    terminated = False
    # 默认就是失败的
    outcome = 0  # 0 for failure, 1 for success

    while not terminated:
        # Choose an action,先生成一个随机数，如果小于epsilon
        # 就随机选择动作，否则就选择Q-table中最大的动作
        # 因为epsilon会逐渐降低，所以一开始会随机选择动作，然后慢慢选择Q-table中最大的动作
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state[0]])
        # Implement this action and move the agent in the desired direction
        observation, reward, terminated, truncated, info = env.step(action)

        # 状态1 observed结果
        q_observed = reward + gamma * np.max(qtable[observation])
        # TD error
        td_error = q_observed - qtable[state[0], action]
        # update Q-table
        qtable[state[0], action] = qtable[state[0], action] + alpha * td_error

        # qtable[state[0], action] = qtable[state[0], action] + \
        #                            alpha * (reward + gamma * np.max(qtable[observation]) - qtable[state[0], action])

        state = [observation]
        if reward == 1:
            outcome = 1
        # update epsilon
        epsilon = max(epsilon - epsilon_decay, 0)

    outcomes.append(outcome)

print('===========================================')
print('Q-table after training:')
print(qtable)

# Plot outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()
