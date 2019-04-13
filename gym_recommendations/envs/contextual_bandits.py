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
         the context matrix is generated only when we create the env , the same for mean_rewards per bandit 
          at each step , if the agent choose an action a , we generate a reward from N (mean_reward[a],sigma ) 
         The action_space : we select k arms from nb_arms  (k = 0.6* nb_arms ) 
         the context size p is set to 3 
    """

        
    metadata = {'render.modes': ['ansi']}

    def __init__(self, nb_arm=10,sigma=0.05):
        self.nb_arm = nb_arm
        self.action = None
        self.state = 0 # stable baseline require some state
        self.info =None ## info for printing 
        self.reward = None
        self.observation_space = None # we will not need it for now
        self.full_actionspace=range(0,self.nb_arm) # total action space
        self.p = 3  ##  t context size
        self.k = int(0.6*self.nb_arm) ## fraction of possible actions to sample for action_space , 60%       
        self.x = random_points_l2_ball(self.nb_arm, self.p).T ## context matrix
        self.action_space =np.random.choice(self.nb_arm, self.k, replace=False) ## initalize to k possible actions
        self.action_space =np.array(self.full_actionspace)[0:self.k] ## initalize to k possible actions
        self.x_context=self.x[self.action_space] ## initialize context
        #self.theta = np.random.uniform(size=(self.p)) #  theta constant for all arms
        self.theta =np.random.uniform(size=(nb_arm,self.p))  # theta with the same shape of x ..
        self.sigma = sigma
        self.mean_reward = np.sum(self.x*self.theta, axis=1)  ## mean reward for each arm  , just the wise dot product of context by theta
        #self.mean_reward = np.dot(self.x, self.theta) ## mean reward for each arm  , just the dot product of
        self.best_action=self.action_space[np.argmax(self.mean_reward [ self.action_space ])] #àsupprimer

    def compute_reward(self)   :
        return np.random.normal(self.mean_reward, self.sigma) [self.action]
    
    def compute_action_space(self) : 
        self.action_space =np.random.choice(self.nb_arm,  self.k, replace=False)  ### generate new action space randomly
        self.x_context=self.x[self.action_space]
        self.best_action=self.action_space[np.argmax(self.mean_reward [ self.action_space ])] ## à supprimer enfin

    def step(self, action):
        #print(action)
        #print(self.action_space)
        assert action in self.action_space
        self.action = action
        self.reward = self.compute_reward()
        self.done= False
        self.compute_action_space()
        return self.state, self.action_space, self.reward, self.done,self.x_context

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.__init__(nb_arm=10)
        return self
    
    def render(self, mode='ansi', close=False):
        outfile = sys.stdout
        inp = "State {}, action {}, reward {}\n".format(self.state, self.action, self.reward)
        outfile.write(inp)
        return outfile

    # def _human_render(self):
    #     plt.plot(self.action, self.reward)
    #     plt.draw()
    #     plt.pause(0.001)s
