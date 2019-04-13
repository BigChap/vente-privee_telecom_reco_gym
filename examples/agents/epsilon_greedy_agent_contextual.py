### optimistic initialisation
import numpy as np

class ContextualEpsilonGreedy:
    def __init__(self, nb_arms,action_space, epsilon=0.1):
        self.action_space = action_space
        self.nb_arms=nb_arms
        self.full_action_space=range(nb_arms)
        self.action_count = {x:0 for x in range(self.nb_arms)}
        self.q_table = {x:100. for x in range(self.nb_arms)}
        self.last_action_space=self.action_space
        self.last_action  =np.random.choice(self.last_action_space)
        self.epsilon = epsilon
        
    def act(self, action_space, x,reward, done):
        self.q_table[self.last_action]=(self.q_table[self.last_action]*self.action_count[self.last_action]+reward)/(self.action_count[self.last_action]+1)
        self.action_count[self.last_action]+=1

        if np.random.random() < self.epsilon: 
            action = np.random.choice(action_space)
        else:
         #   actions = np.array(list(self.q_table.keys()))[np.array(list(self.q_table.values())) == max(list(self.q_table.values()))] 
           # action = np.random.choice(actions)
            temp_q= [list(self.q_table.values())[i] for i in action_space]
            action =action_space[np.argmax(temp_q)]

        self.last_action = action
        self.last_action_space=action_space
        return action

    def reset(self):
        self.__init__(self.action_space,self.epsilon)
        return self
