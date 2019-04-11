import numpy as np

class ThompsonSampling:
    
    def __init__(self, action_space, param=None):
        self.action_space = action_space
        self.cum_rewards = np.zeros(action_space.n,float) 
        self.nb_tries = np.zeros(action_space.n,int)
        self.param = param
        self.last_action = 0
        
    def act(self,observation,reward,done):
        k = np.shape(self.nb_tries)[0]
        self.cum_rewards[self.last_action] += reward
        if self.param == "beta":
            # Beta prior
            try:
                samples = np.random.beta(self.cum_rewards+1, self.nb_tries - self.cum_rewards+1)
            except:
                samples = np.random.random(k)
        else:
            # Normal prior
            samples = np.random.normal(self.cum_rewards / (self.nb_tries+1), 1. / (self.nb_tries+1))
        self.last_action = np.argmax(samples)
        self.nb_tries[self.last_action] += 1
        return self.last_action

    def reset(self):
        self.__init__(action_space=self.action_space, param=self.param)
        return self
