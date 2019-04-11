import numpy as np

class GradientBandit:
    
    def __init__(self, action_space, alpha=0.1):
        # Number of arms
        self.k = action_space.n
        self.actions =np.array(range( action_space.n))
        self.action_space = action_space
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(action_space.n)
        
        # Total mean reward
        self.mean_reward = 0

        # Mean reward for each arm

        self.k_reward = np.zeros(action_space.n)
        # Initialize preferences
        self.H = np.zeros(action_space.n)

        # Learning rate
        self.alpha = alpha
        
        # last action
        self.last_action = 0
        self.prob_action = np.exp(self.H - np.max(self.H)) \
            / np.sum(np.exp(self.H - np.max(self.H)), axis=0)
 
            
    def softmax(self):
        self.prob_action = np.exp(self.H - np.max(self.H)) \
            / np.sum(np.exp(self.H - np.max(self.H)), axis=0)
        

    def act(self,observation,reward,done):

        # Update probabilities
        a=self.last_action
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
                reward - self.k_reward[a]) / self.k_n[a]
        
        # Update total
        self.mean_reward = self.mean_reward + (
                reward - self.mean_reward) / self.n
        
        
        # Update preferences
        self.H[a] = self.H[a] + \
                self.alpha * (reward - self.mean_reward) * (1 - self.prob_action[a])
        
        actions_not_taken = self.actions!=a
        self.H[actions_not_taken] = self.H[actions_not_taken] - \
            self.alpha * (reward - self.mean_reward) * self.prob_action[actions_not_taken]
        
        
         # compute probabilities of each action using softmax
        self.softmax()
        
        # Select highest preference action
        action = np.random.choice(self.actions, p=self.prob_action)
        self.last_action=action
        
        if done : 
            
            # initialize
            self.k_reward = np.zeros(self.k)
            self.mean_reward = 0            
            self.prob_action = np.exp(self.H - np.max(self.H)) \
            / np.sum(np.exp(self.H - np.max(self.H)), axis=0)
 
        return action

    def reset(self):
        self.__init__(self.action_space,self.alpha)
