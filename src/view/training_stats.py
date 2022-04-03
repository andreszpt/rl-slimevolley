from os.path import join
import pandas as pd
from matplotlib import pyplot as plt

LOGDIR = '././logs'
EVAL_FILE = 'progress.csv'

monitor = pd.read_csv(join(LOGDIR, EVAL_FILE), skiprows=1)
episode_rewards = monitor.loc[:, 'rollout/ep_rew_mean']
episode_lengths = monitor.loc[:, 'rollout/ep_len_mean']
episode_times = monitor.loc[:, 'time/time_elapsed']


fig = plt.figure()
ax = plt.axes()
ax.plot(episode_times, episode_rewards)
plt.show()



