import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys



class NonStationaryMAB(gym.Env):
    metadata = {'render.modes': ['ansi']}

    def __init__(self, time_grid, reward_function, mean_reward_functions):
        """

        Args:
            time_grid: list of time points
            mean_reward_functions: list of mean reward functions taking a time point as input
        """
        self.nb_arm = len(mean_reward_functions)
        self.action_space = spaces.Discrete(self.nb_arm)
        self.action = None
        self.reward = None
        self.state = 0  # stable baseline require some state
        self.observation_space = spaces.Discrete(1)
        self.t = None
        self.time_grid = time_grid
        self.reward_function = reward_function
        self.mean_reward_functions = mean_reward_functions

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        self.action = action
        self.reward = self.reward_function(self.mean_reward_functions[action](self.time_grid[self.t]))
        self.t += 1
        return self.state, self.reward, False, dict()  # state, reward, done, info

    def reset(self):
        self.t = 0
        return self.state

    def render(self, mode='ansi', close=False):
        outfile = sys.stdout
        inp = "State {}, action {}, reward {}\n".format(self.state, self.action, self.reward)
        outfile.write(inp)
        return outfile


class NormalNonStationaryMAB(NonStationaryMAB):
    def __init__(self, time_grid, functions):
        def normal(mean):
            return np.random.normal(loc=mean, scale=1, size=1)

        super(NormalNonStationaryMAB, self).__init__(time_grid, normal, functions)


class BernoulliNonStationaryMAB(NonStationaryMAB):
    def __init__(self, time_grid, functions):
        def bernoulli(mean):
            return np.random.binomial(p=mean, n=1, size=1)

        super(BernoulliNonStationaryMAB, self).__init__(time_grid, bernoulli, functions)


class NormalNonStationaryMABv0(NormalNonStationaryMAB):
    def __init__(self):
        time_grid = np.arange(1000)

        def f1(x):
            return 1

        def f2(x):
            if x < 40:
                return 0.
            else:
                return 2.

        functions = [f1, f2]
        super(NormalNonStationaryMABv0, self).__init__(time_grid, functions)


class BernoulliNonStationaryMABv0(BernoulliNonStationaryMAB):
    def __init__(self):
        time_grid = np.arange(1000)

        def f1(x):
            return 0.3

        def f2(x):
            if x < 40:
                return 0.1
            else:
                return 0.6

        functions = [f1, f2]
        super(BernoulliNonStationaryMABv0, self).__init__(time_grid, functions)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    envs = [NormalNonStationaryMABv0(), BernoulliNonStationaryMABv0()]
    for env in envs:
        for i in range(env.nb_arm):
            plt.plot(env.time_grid, [env.mean_reward_functions[i](t) for t in env.time_grid])
            plt.plot(env.time_grid, [env.reward_function(env.mean_reward_functions[i](t)) for t in env.time_grid], 'o', markersize=2)
        plt.show()
