import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C, PPO
import slimevolleygym
from os.path import isfile


class Slime:
    def __init__(self, env, alg):
        self.env = gym.make(env)
        self.alg = alg
        if alg == 'PPO':
            self.model = PPO('MlpPolicy', self.env, verbose=1)
        elif alg == 'A2C':
            self.model = A2C('MlpPolicy', self.env, verbose=1)

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
            action, _state = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
        

class SlimeMonitor(Slime):
    def __init__(self, env, alg):
        self.env = Monitor(env=gym.make(env), filename='./logs/')
        self.alg = alg
        if alg == 'PPO':
            self.model = PPO('MlpPolicy', self.env, verbose=1)
        elif alg == 'A2C':
            self.model = A2C('MlpPolicy', self.env, verbose=1)
    
    