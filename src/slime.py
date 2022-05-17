from re import S
import gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import A2C, PPO, DQN
from baseline import BaselinePolicy
import slimevolleygym
from os.path import isfile, join
import numpy as np

LOGDIR = join('..', 'logs')

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
        eval_callback = EvalCallback(
            eval_env=self.env,
            best_model_save_path=LOGDIR,
            log_path=LOGDIR,
            eval_freq=1000)
        self.model.learn(total_timesteps=t, callback=eval_callback)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        if(isfile(path)):
            if self.alg == 'PPO':
                self.model = PPO.load(path)
            elif self.alg == 'A2C':
                self.model = A2C.load(path)
            elif self.alg == 'DQN':
                self.model = DQN.load(path)

    def simulate(self):
        obs = self.env.reset()
        while True:
            action, _states = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            
    def sample_states(self, n_samples, mode):
        obs = self.env.reset()
        sampled_states = np.empty((n_samples, 12))
        for i in range(n_samples):
            if (mode == 'RANDOM'):
                action = self.env.action_space.sample()
            elif (mode == 'MODEL'):
                action = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            if (i % 1000 == 0):
                print(f'Sample n = {i}')
            sampled_states[i] = info["state"]
        return sampled_states