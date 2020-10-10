import gym

import rlephant

env = gym.make('CartPole-v0')
env.reset()
storage = rlephant.ReplayStorage('cartpole.h5')
episode = rlephant.Episode()

action = env.action_space.sample()
obs, reward, done, info = env.step(action)

transition = rlephant.Transition(
    observation={'some_obs': obs},
    action={'some_action': action},
    reward=reward,
    done=done)

episode.append(transition)

storage.save(episode)

last_episode = storage[-1]
for transition in last_episode:
    print(transition)
