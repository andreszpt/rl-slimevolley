from slime import Slime
from gym.spaces import Discrete



if __name__ == '__main__':
    alg_list = ['A2C', 'DQN', 'PPO']        # TRPO and QR-DQN belong to SB3 Contrib
    selected_alg = 'PPO'
    mode = 'LOAD'
    timestamps = 20_000

    if mode == 'TRAIN':
        for alg in alg_list:
            slime = Slime(alg)
            slime.train(timestamps)
            slime.save_model(f'./models/{alg}_{timestamps//1000}K')

    if mode == 'LOAD':
        slime = Slime(selected_alg)
        slime.load_model('./models/PPO_20K')
        slime.simulate()
