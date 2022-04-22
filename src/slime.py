from re import S
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C, PPO, DQN
from baseline import BaselinePolicy
import slimevolleygym
from os.path import isfile
import numpy as np


class Slime:
    env = gym.make('SlimeVolley-v0')

    def __init__(self, alg):
        self.alg = alg
        if alg == 'PPO':
            self.model = PPO('MlpPolicy', self.env, verbose=1)
        elif alg == 'A2C':
            self.model = A2C('MlpPolicy', self.env, verbose=1)
        elif alg == 'DQN':
            self.model = DQN('MlpPolicy', self.env, verbose=1)
        elif alg == 'BSLN':
            self.model = BaselinePolicy()

    def train(self, t):
        self.model.learn(total_timesteps=t)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        if(isfile(path)):
            if self.alg == 'PPO':
                self.model = PPO.load(path)
            elif self.alg == 'A2C':
                self.model = A2C.load(path)

    def simulate(self):
        obs = self.env.reset()
        while True:
            action = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            
    def sample_states(self, n_samples):
        obs = self.env.reset()
        sampled_states = np.empty((n_samples, 12))
        for i in range(n_samples):
            action = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            if (i % 10 == 0):
                print(f'Sample n = {i}')
            sampled_states[i] = info["state"]
        return sampled_states