import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,'..')
print(sys.path)
from gym_recommendations.envs.contextual_bandits import *  ### import contextual env


    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    #outdir = '/tmp/random-agent-results'
    #video_callable = None if render_mode == 'human' else False

    #env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=video_callable)
    #env = DynamicMonitor(env, directory=outdir, force=True, video_callable=video_callable)
from gym import wrappers, logger
from agents.random_agent_contextual import RandomAgentContext
from agents.epsilon_greedy_agent_contextual import EpsilonGreedyContext
#from gym_recommendations.wrapper import DynamicMonitor


import click

@click.command()
@click.option('--env_name', default='Multi-Armed-Bandits-v0', help='Select the environment to run.')
@click.option('--agent_name', default='EpsilonGreedyAgentContext', help='Select the agent to run.')
@click.option('--nb_episodes', default=3, help='Number of episodes to run.')
@click.option('--max_episode_steps', default=100, help='Number of max_episode_steps to run.')
@click.option('--render_freq', default=1, help='Rendering frequency.')
@click.option('--render_mode', default='human', help='Randering mode.')
@click.option('--nb_arm', default=10, help='select number of arms')
def run(env_name, agent_name, nb_episodes,max_episode_steps, render_freq, render_mode,nb_arm):
    logger.set_level(logger.INFO)
    
    

    env =ContextualBandit(nb_arm)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    #outdir = '/tmp/random-agent-results'
    #video_callable = None if render_mode == 'human' else False

    #env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=video_callable)
    #env = DynamicMonitor(env, directory=outdir, force=True, video_callable=video_callable)

    env.render(mode=render_mode)
    env.seed(0)
    
    if agent_name == 'RandomAgentContext' :
        agent  =RandomAgentContext(env.action_space)
    elif agent_name == 'EpsilonGreedyAgentContext' :
        agent = EpsilonGreedyContext((np.array(range(nb_arm))))
    
   

    step = 0
    reward = 0
    done = False

    for episode in range(nb_episodes):
        print(f'--------- Episode {episode} ---------')
        ob = env.reset()
        step = 0
        actions_iter=env.action_space

        while step <= max_episode_steps:
            step += 1
            #print(step)
            # action space may have change
            # agent = EpsilonGreedy(env.env.acenvcotion_space)            
            action = agent.act(actions_iter,ob, reward, done)
            state, actions_iter, reward, done ,X= env.step(action)
            if done:
                break
            if step % render_freq == 0:
                env.render()
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not rec
            
           
    # Close the env and write monitor result info to disk
    #env.env.close()

if __name__ == '__main__':
    run()
