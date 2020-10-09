
from time import sleep
import gym
import racecar_gym

from elephant.collector import DataCollector, Episode, DataReader

env = gym.make('MultiAgentAustria-v0')

done = False
obs = env.reset()

reader = DataReader('A.hdf5')
episode = reader.episode(index=1)

collectors = None
#collectors = dict((agent, DataCollector(directory=agent, batch_size=5)) for agent in 'A,B,C,D'.split(','))

for i in range(10):
    episode = dict((agent, Episode()) for agent in 'A,B,C,D'.split(','))
    env.reset()
    done = True
    while not done:
        action = env.action_space.sample()
        obs, rewards, dones, states = env.step(action)
        done = any(dones.values())
        for agent in obs:
            episode[agent].append((obs[agent], action[agent], rewards[agent], dones[agent], states[agent]))

    for agent in obs:
        collectors[agent].save_episode(episode[agent])

env.close()