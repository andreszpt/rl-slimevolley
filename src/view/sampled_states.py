from os.path import join
import numpy as np
from matplotlib import pyplot as plt

LOGDIR = '././logs'
SAMPLES_NPY_FILE = '1_Ksamples.npy'

sampled_states = np.load(join(LOGDIR, SAMPLES_NPY_FILE))

agent_x_pos = sampled_states[:, 0]
agent_y_pos = sampled_states[:, 1]
agent_x_sp = sampled_states[:, 2]
agent_y_sp = sampled_states[:, 3]

fig = plt.figure()
ax = plt.axes()
ax.scatter(agent_x_pos, agent_y_pos, c='g')
ax.scatter(agent_x_sp, agent_y_sp, c='r')
plt.show()