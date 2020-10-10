import gym

import rlephant

env = gym.make('CartPole-v0')

storage = rlephant.ReplayStorage('cartpole.h5')

episodes = 5
for _ in range(episodes):
    episode = rlephant.Episode()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        episode.append(rlephant.Transition(
            observation={'obs': obs},
            action={'action': action},
            reward=reward,
            done=done
        ))
    storage.save(episode)

storage = rlephant.ReplayStorage('cartpole.h5')
print(f'Episodes: {len(storage)}')

for i, episode in enumerate(storage[-5:]):
    print(f'Episode {i}: length={len(episode)}, rewards={sum(episode.rewards)}')
