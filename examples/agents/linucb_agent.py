import numpy as np

class LinearUcb:

    def __init__(self, nb_arms,action_space,context, alpha=1.0):
        self.action_space = action_space
        self.nb_arms=nb_arms
        self.full_action_space=range(nb_arms)
        self.d=context.shape[1]
        self.alpha = float(alpha)
        self.A=[np.identity(self.d)]*len(self.full_action_space)
        self.b=[np.zeros((self.d))]*len(self.full_action_space)
        self.last_action_space=action_space ## un actionspace réçu initialement
        self.last_action=np.random.choice(self.last_action_space) ##  une action aléa
        self.last_context=context
        self.payoff=np.zeros(nb_arms)
        self.tested_arms=[0] ##  for exploration
        self.n=0
        #self.payoff=np.ones(nb_arms)*-1#à supprimer

        #### for reset ####
        self.init_context=context
        self.init_action_space=action_space
    def act(self, action_space, x,reward, done):
        
        ########  update A and b for the last action #######
        a = self.last_action        
        xa_last =  np.hstack(self.last_context[np.where(self.last_action_space==a)])
        self.A[a]+=  xa_last@xa_last.T
        self.b[a]+=reward*xa_last.T
        
        
        
        ######## loop all actions                ######
        for action in action_space : 
            xa =  np.hstack(x[np.where(action_space==action)])
            theta_a= np.linalg.inv(self.A[action])@self.b[action]
           #print( theta_a)
           # print(xa.T)
           # print(self.A[action])
            self.payoff[action]=xa.T@theta_a + self.alpha * \
            np.sqrt(xa.T@np.linalg.inv(self.A[action])@xa)
        
        self.last_action =action_space[np.argmax(self.payoff [action_space])]
        
        if self.n < self.nb_arms+10: 
            cand=[x for x in action_space if x not in self.tested_arms]
            if len(cand)>0 :
                self.last_action=np.random.choice(cand)
                self.tested_arms=self.tested_arms+[self.last_action]
                self.n+=1
        self.last_action_space=action_space
        self.last_context=x
        
        return self.last_action

    def reset(self):
        self.__init__(self.nb_arms,self.init_action_space,self.init_context,self.alpha )
        return self
