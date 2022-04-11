from os import path
import pandas as pd
from matplotlib import pyplot as plt

DATADIR = path.join('..', 'data')
SAMPLES_FILE = 'sampled_states.csv'

sampled_states = pd.read_csv(path.join(DATADIR, SAMPLES_FILE))

agent_x_pos = sampled_states[:, 0]
agent_y_pos = sampled_states[:, 1]
agent_x_sp = sampled_states[:, 2]
agent_y_sp = sampled_states[:, 3]

fig = plt.figure()
ax = plt.axes()
ax.scatter(agent_x_pos, agent_y_pos, c='g', s=1)
ax.scatter(agent_x_sp, agent_y_sp, c='r', s=1)
plt.show()