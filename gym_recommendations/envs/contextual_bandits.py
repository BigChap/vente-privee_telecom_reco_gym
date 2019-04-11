import gym
import numpy as np
import sys
from gym.utils import seeding
from tools import *



##########################  contextual bandit environement ###############################


class ContextualBandit():
    # Contextual MAB
    
    """
    Many assumptions :
         the context matrix is generated only when we creat the env , the same for mean_rewards per bandit 
          at each step , if the agent choose an action a , we generate a reward from N (mean_reward[a],sigma ) 
         The action_space : we select k arms from nb_arms  (k = 0.5 * nb_arms ) 
         the context size p is set to 3 
    """

        
    metadata = {'render.modes': ['ansi']}

    def __init__(self, nb_arm=10):
        self.nb_arm = nb_arm
        self.action = None
        self.state = 0 # stable baseline require some state
        self.info =None ## info for printing 
        self.reward = None
        self.observation_space = None # we will not need it for now
        self.p = 3  ##  t context size
        self.k = int(0.5*self.nb_arm) ## fraction of possible actions to sample for action_space , 50%
        self.action_space =np.random.choice(self.nb_arm, self.k, replace=False) ## initalize to k possible actions
        self.x = random_points_l2_ball(self.nb_arm, self.p).T ## context matrix
        self.theta = np.random.uniform(size=(self.p))  
        self.sigma = 0.05
        self.mean_reward = np.dot(self.x, self.theta) ## mean reward for each arm  , just the dot product of context by theta

    def compute_reward(self)   :
        return np.random.normal(self.mean_reward, self.sigma) [self.action]
    
    def compute_action_space(self) : 
         return np.random.choice(self.nb_arm,  self.k, replace=False)  ### generate new action space randomly
    
    def step(self, action):
        assert action in self.action_space
        self.action = action
        self.reward = self.compute_reward()
        self.done= False
        self.action_space = self.compute_action_space()
        return self.state, self.action_space, self.reward, self.done,self.x[self.action_space]
    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        
        self.x = random_points_l2_ball(self.nb_arm, self.p).T ## contet matrix
        self.theta = np.random.uniform(size=(self.p)) 
        self.sigma = 0.05
        self.mean_reward = np.dot(self.x, self.theta) ## mean reward for each arm
        return self.state

    def render(self, mode='ansi', close=False):
        outfile = sys.stdout
        inp = "State {}, action {}, reward {}\n".format(self.state, self.action, self.reward)
        outfile.write(inp)
        return outfile

    # def _human_render(self):
    #     plt.plot(self.action, self.reward)
    #     plt.draw()
    #     plt.pause(0.001)
