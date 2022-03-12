from slime import Slime
from gym.spaces import Discrete



if __name__ == '__main__':
    alg_list = ['A2C', 'DQN', 'PPO']        # TRPO and QR-DQN belong to SB3 Contrib
    mode = input(
        f'''Please, select one of the following options:
        "T": Train the agent this set of algorithms: {alg_list}
        "L": Load an algorithm from the given list and start the simulation
        (T/L): '''
        )
    if (mode == 'L'):
        selected_alg = input(
            f'''Select an algorithm from the list: {alg_list}: \n'''
        )
    n_timesteps = 20_000

    if mode == 'T':
        for alg in alg_list:
            slime = Slime(alg)
            slime.train(n_timesteps)
            slime.save_model(f'./models/{alg}_{n_timesteps//1000}K')

    if mode == 'L':
        slime = Slime(selected_alg)
        slime.load_model('./models/PPO_20K')
        slime.simulate()
