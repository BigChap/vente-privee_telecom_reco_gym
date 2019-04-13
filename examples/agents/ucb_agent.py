import numpy as np

class ucb:

    def __init__(self, action_space, c=1.0):
        self.action_space = action_space
        self.c = float(c)
        # Number of arms
        self.k = action_space.n
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(action_space.n)
        # Total mean reward
        self.mean_reward = 0
        # Mean reward for each arm
        self.k_reward = np.ones(action_space.n)*100
        self.last_action = 0

    def act(self, observation, reward, done):
        if self.n==0:
            action = np.random.randint(self.action_space.n)
        else:
            # Update counts
            self.n += 1
            self.k_n[self.last_action] += 1

            # Update total
            self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

            # Update results for a_k
            self.k_reward[self.last_action] = self.k_reward[self.last_action] + (
                    reward - self.k_reward[self.last_action]) / self.k_n[self.last_action]

            action = np.argmax(self.k_reward + self.c * np.sqrt((np.log(self.n)) / self.k_n))
        
        self.last_action = action

        return action

    def reset(self):
        self.__init__(self.action_space,self.c)
        return self

