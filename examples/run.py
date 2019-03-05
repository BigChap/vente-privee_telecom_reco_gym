import argparse
import sys
import gym_recommendations
import gym
from gym import wrappers, logger
from agents.random_agent import RandomAgent
#from gym_recommendations.wrapper import DynamicMonitor


import click

@click.command()
@click.option('--env_name', default='Multi-Armed-Bandits-v0', help='Select the environment to run.')
@click.option('--agent_name', default='RandomAgent', help='Select the agent to run.')
@click.option('--nb_episodes', default=3, help='Number of episodes to run.')
@click.option('--render_freq', default=1, help='Rendering frequency.')
@click.option('--render_mode', default='human', help='Randering mode.')
def run(env_name, agent_name, nb_episodes, render_freq, render_mode):
    logger.set_level(logger.INFO)

    env = gym.make(env_name)

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
    #agent = RandomAgent(env.env.action_space)

    step = 0
    reward = 0
    done = False

    for episode in range(nb_episodes):
        print(f'--------- Episode {episode} ---------')
        ob = env.reset()
        while True:
            step += 1
            # action space may have change
            agent = RandomAgent(env.env.action_space)
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            if step % render_freq == 0:
                env.render()
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.env.close()

if __name__ == '__main__':
    run()
