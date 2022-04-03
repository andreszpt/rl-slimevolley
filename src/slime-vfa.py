from os.path import join
from slime import SlimeVFA
import numpy as np


if __name__ == '__main__':
    N_SAMPLES = 1_000
    LOGDIR = './logs'
    OUTPUT_FILE = str(N_SAMPLES//1000)+'K_samples'
        
    slime_vfa = SlimeVFA()
    sampled_states = slime_vfa.simulate(N_SAMPLES)
    np.save(join(LOGDIR, OUTPUT_FILE), sampled_states)