import gym
import slimevolleygym
from os.path import join
from stable_baselines3 import PPO, SAC
from gym.wrappers import Monitor

VIDDIR = join('..', 'videos')

class ReducedDimension(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = gym.spaces.Box(shape=(8,), low=-2.0, high=2.0)
    def observation(self, obs):
        return obs[0:8]


# Simulation using random actions

env = Monitor(gym.make('SlimeVolley-v0'), './videos/random', force=True)
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
env.close()


# Simulation using PPO

env_ppo = Monitor(ReducedDimension(gym.make('SlimeVolley-v0')), './videos/ppo', force=True)
model = PPO.load('models\PPO_WRA\PPO_3M_WRA.zip')
obs = env_ppo.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env_ppo.step(action)
    if done:
        obs = env_ppo.reset()
env_ppo.close()




