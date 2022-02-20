from slime import Slime


if __name__ == '__main__':
    slime = Slime('SlimeVolley-v0', 'PPO')
    slime.load_model('./models/ppo_slime_250k')
    slime.train(20_000)
    # slime.save_model('./models/ppo_slime_250k')
    slime.simulate()
