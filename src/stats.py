from os import times
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

fig = plt.figure()
ax = plt.axes()

monitor = pd.read_csv('./logs/monitor.csv', skiprows=1)
episode_rewards = monitor.loc[:, 'r']
episode_lengths = monitor.loc[:, 'l']
episode_times = monitor.loc[:, 't']

# ax.plot(episode_times, episode_rewards)
# plt.show()


sampled_states = np.load('./logs/1Ksample_20Kt.npy')
agent_x_pos = sampled_states[:, 0]
agent_y_pos = sampled_states[:, 1]
agent_x_sp = sampled_states[:, 2]
agent_y_sp = sampled_states[:, 3]


ax.scatter(agent_x_pos, agent_y_pos, c='g')
ax.scatter(agent_x_sp, agent_y_sp, c='r')
plt.show()
