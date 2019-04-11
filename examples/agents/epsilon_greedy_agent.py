### optimistic initialisation
import numpy as np

class EpsilonGreedy:
    def __init__(self, action_space, epsilon=0.1):
        self.action_space = action_space
        self.action_count = {x:0 for x in range(action_space.n)}
        self.q_table = {x:100. for x in range(action_space.n)}
        self.last_action = 0
        self.epsilon = epsilon

    def act(self, observation, reward, done):
        self.q_table[self.last_action]=(self.q_table[self.last_action]*self.action_count[self.last_action]+reward)/(self.action_count[self.last_action]+1)
        self.action_count[self.last_action]+=1

        if np.random.random() < self.epsilon: 
            action = np.random.randint(self.action_space.n) 
        else:
            actions = np.array(list(self.q_table.keys()))[np.array(list(self.q_table.values())) == max(list(self.q_table.values()))] 
            action = np.random.choice(actions)

        self.last_action = action
        return action

    def reset(self):
        self.__init__(self.action_space,self.epsilon)
        return self
