### optimistic initialisation
import numpy as np

class EpsilonGreedyContext:
    def __init__(self, action_space,epsilon=0.1,steps=30):
        self.action_space = action_space
        self.action_count = {x:0 for x in range(action_space.size)}
        self.q_table = {x:100. for x in range(action_space.size)}
        self.last_action = 0
        self.epsilon = epsilon
        self.exploration_step = steps

    def act(self, action_space,observation, reward, done):
        if done:
            self.action_count = {x:0 for x in range(self.action_space.size)}
            self.q_table = {x:100. for x in range(self.action_space.size)}
        if not done:
            self.q_table[self.last_action]=(self.q_table[self.last_action]*self.action_count[self.last_action]+reward)/(self.action_count[self.last_action]+1)
            self.action_count[self.last_action]+=1

        if (np.random.random() < self.epsilon or sum(list(self.action_count.values()))<self.exploration_step):
            action = np.random.choice(action_space) 
        else:
            #action = list(self.q_table.keys()[action_space])[list(self.q_table[action_space].values()).index(max(self.q_table[action_space].values()))] 
            temp_q= [list(self.q_table.values())[i] for i in action_space]
            action =action_space[np.argmax(temp_q)]
        self.last_action = action
        return action
