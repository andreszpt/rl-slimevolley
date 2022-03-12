from slime import SlimeVFA
import numpy as np


if __name__ == '__main__':
    alg_list = ['A2C', 'DQN', 'PPO']        # TRPO and QR-DQN belong to SB3 Contrib
    mode = input(
        f'''Please, select one of the following options:
        "T": Train the agent with this set of algorithms: {alg_list}
        "L": Load an algorithm from the given list and start the simulation
        (T/L): '''
        )
    if (mode == 'L'):
        selected_alg = input(
            f'''Select an algorithm from the list: {alg_list}: \n'''
        )
    n_timesteps = 20_000
    n_samples = 1_000

    if mode == 'T':
        for alg in alg_list:
            slime_vfa = SlimeVFA(alg)
            slime_vfa.train(n_timesteps)
            slime_vfa.save_model(f'./models/{alg}_{n_timesteps//1000}K')

    if mode == 'L':
        slime_vfa = SlimeVFA(selected_alg)
        slime_vfa.load_model('./models/PPO_50K')
        sampled_states = slime_vfa.simulate(n_samples)
        np.save(
        f'./logs/{n_samples//1000}Ksample_{n_timesteps//1000}Kt', sampled_states)