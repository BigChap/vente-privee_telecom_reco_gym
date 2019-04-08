import numpy as np 
class RandomAgentContext(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self,action_space, observation, reward, done):
        return np.random.choice(action_space)