from numpy import savetxt
from slime import Slime
from os.path import join


if __name__ == '__main__':
    alg_list = ['A2C', 'DQN', 'PPO']        # TRPO and QR-DQN belong to SB3 Contrib
    N_TIMESTEPS = 20_000
    N_SAMPLES = 1_000_000
    DATADIR = join('..', 'data')
    MODELDIR = join('..', 'models')
    UTILDIR = join('..', 'utils')

    # TODO: Do not use default N_TIMESTEPS and N_SAMPLES
    mode = input(
        f'''Please, select one of the following options:
        "T": Train the agent this set of algorithms: {alg_list} with default n_timesteps (20K)
        "L": Load an algorithm from the given list and start the simulation
        "B1": Simulate a game where both slimes use Baseline policy
        "B2": Sample default n_states (1M) and export into data/sampled_states.csv
        (T/L/B1/B2): '''
        )
    
    if mode == 'L':
        selected_alg = input(
            f'''Select an algorithm from the list: {alg_list}: \n'''
        )
        slime = Slime(selected_alg)
        slime.load_model(join(MODELDIR, f'{selected_alg}_20K'))
        slime.simulate()

    if mode == 'T':
        for alg in alg_list:
            slime = Slime(alg)
            slime.train(N_TIMESTEPS)
            slime.save_model(join(MODELDIR, f'{alg}_{N_TIMESTEPS//1000}K'))
    
    if mode == 'B1':
        slime = Slime('BSLN')
        slime.simulate()
        
    if mode == 'B2':
        slime = Slime('BSLN')
        sampled_states = slime.sample_states(N_SAMPLES, 'MODEL')
        head = 'x_agent,y_agent,xdot_agent,ydot_agent,' \
            'x_ball,y_ball,xdot_ball,ydot_ball,' \
            'x_opponent,y_opponent,xdot_opponent,ydot_opponent'
        savetxt(fname=join(DATADIR, 'sampled_states.csv'),
                X=sampled_states,
                fmt='%.5f',
                delimiter=',',
                header=head,
                comments='')
        