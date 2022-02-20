from os import times
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

monitor = pd.read_csv('./logs/monitor.csv', skiprows=1)
episode_rewards = monitor.loc[:, 'r']
episode_lengths = monitor.loc[:, 'l']
episode_times = monitor.loc[:, 't']

plt.plot(episode_times, episode_rewards)
plt.show()