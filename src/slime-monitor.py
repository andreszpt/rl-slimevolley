from slime import Slime, SlimeMonitor

if __name__ == '__main__':
    slime = SlimeMonitor('SlimeVolley-v0', 'PPO')
    slime.train(5_000)
    slime.simulate()

