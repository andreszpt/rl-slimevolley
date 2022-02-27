from slime import Slime, SlimeMonitor

if __name__ == '__main__':
    slime = SlimeMonitor('PPO')
    slime.train(10_000)
    slime.simulate()

