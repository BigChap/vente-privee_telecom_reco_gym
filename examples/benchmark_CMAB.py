import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,'..')
print(sys.path)
from gym_recommendations.envs.contextual_bandits import *  ### import contextual env
from gym import wrappers, logger
from agents.random_agent_contextual import RandomAgentContext
from agents.epsilon_greedy_agent_contextual import EpsilonGreedyContext
#from gym_recommendations.wrapper import DynamicMonitor


nb_exp=50
nb_episodes=10
nb_arm=5
max_episode_steps=100


env_name = 'Multi-Armed-Contextual-Bandits'
env =ContextualBandit(nb_arm)




agents_list={'Random Agent':RandomAgentContext(env.action_space),'Epsilon Greedy Agent':EpsilonGreedyContext((np.array(range(nb_arm))))}

def run_bench():

    rewards={x:[] for x in list(agents_list.keys())}
    for i in range(nb_exp):
        print(f'exp {i}')
        for agent_name in list(agents_list.keys()):
            #print(agent_name)
            agent = agents_list[agent_name]
            for episode in range(nb_episodes):
                step=0
                ob = None
                reward=0
                done=False
                reward_record = [0]
            
                
                
                
                
                
                actions_iter=env.action_space
                while step <= max_episode_steps:
                    step += 1
                    #print(action_iter)
                    action = agent.act(actions_iter,ob, reward, done)
                    state, actions_iter, reward, done,X = env.step(action)
                    if done:
                        break
                    reward_record.append(reward_record[max(0,step-1)]+reward)
            rewards[agent_name].append(reward_record)

    #env.env.close()

    fig = plt.figure()
    for agent_name in rewards:
        x = np.mean(rewards[agent_name],axis=0)
        plt.plot(x,label=f'{agent_name}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_bench()


