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
                'Thompson Sampling Agent':ThompsonSampling(env.env.action_space, param="normal")}

def run_bench():
    logger.set_level(logger.INFO)

    rewards={x:[] for x in list(agents_list.keys())}

    for i in range(nb_exp):
        print(f'exp {i}')
        for agent_name in list(agents_list.keys()):
            agent = agents_list[agent_name]
            for _ in range(nb_episodes):
                step=0
                ob = env.reset()
                reward=0
                done=False
                reward_record = [0]
                while True:
                    step += 1
                    action = agent.act(ob,reward,done)
                    ob, reward, done, _ = env.step(action)
                    if done:
                        break
                    reward_record.append(reward_record[step-1]+reward)
                rewards[agent_name].append(reward_record)

    env.env.close()

    plt.figure()
    for agent_name in rewards:
        x = np.mean(rewards[agent_name],axis=0)
        plt.plot(x,label=f'{agent_name}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_bench()


