from gym.envs.registration import register

register(
    id='Multi-Armed-Bandits-v0',
    entry_point='gym_recommendations.envs:MultiArmedBandit',
    max_episode_steps=1000,
)

register(
    id='Bernoulli-NonStationary-MAB-v0',
    entry_point='gym_recommendations.envs:BernoulliNonStationaryMABv0',
    max_episode_steps=200,
)

register(
    id='Normal-NonStationary-MAB-v0',
    entry_point='gym_recommendations.envs:NormalNonStationaryMABv0',
    max_episode_steps=200,
)
