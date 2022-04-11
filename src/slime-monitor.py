from slime import Slime, SlimeMonitor

# TODO: Define what metrics to analyze
if __name__ == '__main__':
    slime = SlimeMonitor('PPO')
    slime.train(10_000)
    slime.simulate()

