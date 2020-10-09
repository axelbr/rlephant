import gym

from elephant import ReplayStorage, Episode, Transition


class StorageWrapper(gym.Wrapper):

    def __init__(self, env, filename: str, batch_size: int = None, max_steps: int = None):
        super().__init__(env)
        self._storage = ReplayStorage(filename=filename, batch_size=batch_size, max_steps=max_steps)
        self._episode = Episode()

    def step(self, action):
        obs, rewards, done, states = self.env.step(action)
        transition = Transition(observation=obs,
                                action=action,
                                reward=rewards,
                                done=done)
        self._episode.append(transition)

    def reset(self, **kwargs):
        obs = self.env.reset()
        self._storage.save(episode=self._episode)
        return obs

    def close(self):
        self._storage.flush()
        self.env.close()