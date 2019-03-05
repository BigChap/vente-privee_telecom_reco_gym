from gym.envs.registration import register

register(
    id='Multi-Armed-Bandits-v0',
    entry_point='gym_recommendations.envs:MultiArmedBandit',
    max_episode_steps=200,
)

