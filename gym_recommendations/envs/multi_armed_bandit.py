import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
#import matplotlib.pyplot as plt


class MultiArmedBandit(gym.Env):
    # Classic Normal MAB from Sutton.
    metadata = {'render.modes': ['ansi']}

    def __init__(self, nb_arm=10):
        self.nb_arm = nb_arm
        self.action_space = spaces.Discrete(self.nb_arm)
        self.arms_mean = np.random.normal(0, 1, self.nb_arm)
        self.action = None
        self.state = 0 # stable baseline require some state
        self.reward = None
        self.observation_space = spaces.Discrete(1)
        self.best_reward = max(self.arms_mean)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        self.action = action
        self.reward = np.random.normal(self.arms_mean[action], 1)
        return self.state, self.reward, False, dict()  # state, reward, done, info

    def reset(self):
        self.arms_mean = np.random.normal(0, 1, self.nb_arm)
        return self.state
    
    def get_best_reward(self):
        return self.best_reward
    
    def render(self, mode='ansi', close=False):
        outfile = sys.stdout
        inp = "State {}, action {}, reward {}\n".format(self.state, self.action, self.reward)
        outfile.write(inp)
        return outfile

    # def _human_render(self):
    #     plt.plot(self.action, self.reward)
    #     plt.draw()
    #     plt.pause(0.001)
