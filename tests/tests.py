import os
from time import sleep
import gym
import racecar_gym

from elephant.persistence import Episode, Transition, ReplayStorage

env = gym.make('MultiAgentAustria-v0')

done = False
obs = env.reset()


os.remove('A.hdf5')
os.remove('B.hdf5')
os.remove('C.hdf5')
os.remove('D.hdf5')


storages = dict((agent, ReplayStorage(filename=agent+'.hdf5', batch_size=3, max_steps=1000)) for agent in 'A,B,C,D'.split(','))


for i in range(5):
    episode = dict((agent, Episode()) for agent in 'A,B,C,D'.split(','))
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, rewards, dones, states = env.step(action)
        done = any(dones.values())
        for agent in obs:
            transition = Transition(observation=obs[agent],
                                    action=action[agent],
                                    reward=rewards[agent],
                                    done=dones[agent])

            episode[agent].append(transition)

    for agent in obs:
        print(episode[agent].length)
        storages[agent].save(episode[agent])

for s in storages.values():
    s.flush()


for e in storages['A'][2:4]:
    print(e.length)
    for t in e[10:20]:
        print(t)

env.close()