import gym_recommendations
import gym
import unittest

def test_random_action():
    env = gym.make('Multi-Armed-Bandits-v0')
    env.reset()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    assert state is None
    assert done is False
