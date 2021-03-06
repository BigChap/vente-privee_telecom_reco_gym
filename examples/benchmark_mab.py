import argparse
import gym_recommendations
import gym
from gym import wrappers, logger
from agents.random_agent import RandomAgent
from agents.epsilon_greedy_agent import EpsilonGreedy
from agents.gradient_bandit_agent import GradientBandit
from agents.thompson_sampling import ThompsonSampling
from agents.ucb_agent import ucb
import numpy as np
import matplotlib.pyplot as plt
import sys


nb_exp=10
nb_episodes=10
env_name = 'Multi-Armed-Bandits-v0'

if len(sys.argv)>1:
    nb_exp = int(sys.argv[1])
if len(sys.argv)>2:
    nb_episodes = int(sys.argv[2])
if len(sys.argv)>3:
    env_name = sys.argv[3] 

env = gym.make(env_name)
agents_list={'Random Agent':RandomAgent(env.env.action_space),\
    'Epsilon Greedy Agent':EpsilonGreedy(env.env.action_space),\
    'Gradient Bandit Agent':GradientBandit(env.env.action_space),\
    'UCB Agent':ucb(env.env.action_space),\
    'Thompson Sampling Agent':ThompsonSampling(env.env.action_space)}

def run_bench():
    logger.set_level(logger.INFO)

    rewards={x:[] for x in list(agents_list.keys())}
    regrets={x:[] for x in list(agents_list.keys())}
    brs={x:[] for x in list(agents_list.keys())}
    for i in range(nb_exp):
        print(f'exp {i}')
        for _ in range(nb_episodes):
            env.env.reset()
            for agent_name in list(agents_list.keys()):
                agent = agents_list[agent_name]
                #print(agent.reset())
                agent = agent.reset()
                ob = env.reset()
                step = 0
                reward=0
                reward_record = [0]
                regret_record = [0]
                br_record = [0]
                done = False
                while True:
                    step += 1
                    action = agent.act(ob,reward,done)
                    ob, reward, done, _ = env.step(action)
                    # print(f'{agent_name} - action:{action} - reward:{reward}')
                    best_reward = env.env.get_best_reward()
                    if done:
                        break
                    reward_record.append(reward_record[step-1]+reward)
                    regret_record.append(regret_record[step-1]+best_reward-reward)
                    br_record.append(best_reward)
                rewards[agent_name].append(reward_record)
                regrets[agent_name].append(regret_record)
                brs[agent_name].append(br_record)
    env.env.close()

    plt.figure()
    for agent_name in rewards:
        x = np.mean(rewards[agent_name],axis=0)
        plt.plot(x,label=f'{agent_name}')
    
    plt.title('Score Cumulé')
    plt.legend()
    plt.savefig(env_name+'_Score')
    plt.show()
    
    f1 = plt.figure()
    for agent_name in regrets :
        x = np.mean(regrets[agent_name],axis=0)
        plt.plot(x,label=f'{agent_name}')
    plt.title('Regret Cumulé')
    plt.legend()
    plt.savefig(env_name+'_Regret')
    plt.show()

    plt.figure()
    plt.plot(np.mean(brs['Random Agent'],axis=0))
    plt.title('Meilleure Récompense')
    plt.savefig(env_name+'_Best_Reward')
    plt.show()

if __name__ == '__main__':
    run_bench()

