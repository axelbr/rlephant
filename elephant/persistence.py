from typing import List

import h5py

from elephant.entities import Episode


class ReplayStorage:

    def __init__(self, filename: str, batch_size: int = None, max_steps: int = None):
        self._batch_size = batch_size if batch_size else 1
        self._max_steps = max_steps
        self._batch = []
        self._file = h5py.File(filename, mode='a')
        self._episodes = 0
        self._keys = ['obs', 'action']

    def _get_item(self, index: int):
        index = index if index >= 0 else len(self._file) + index
        episode_data = self._file[str(index)]
        obs, action = {}, {}

        for k in episode_data['obs']:
            obs[k] = episode_data['obs'][k][:]

        for k in episode_data['action']:
            action[k] = episode_data['action'][k][:]

        rewards = episode_data['reward'][:]
        done = episode_data['done'][:]
        return Episode(observations=obs, actions=action, rewards=rewards, done=done)

    def _get_slice(self, item: slice):
        start = item.start if item.start else 0
        stop = item.stop if item.stop else len(self._file)
        step = item.step if item.step else 1

        if start < 0:
            start = len(self._file) + start

        if stop < 0:
            stop = len(self._file) + stop

        if step < 0:
            tmp = start
            start = stop - 1
            stop = tmp - 1

        episodes = []
        for i in range(start, stop, step):
            episode = self._get_item(i)
            episodes.append(episode)

        return episodes

    def __len__(self):
        return len(self._file)

    def __getitem__(self, item):
        if type(item) == int:
            return self._get_item(item)
        elif type(item) == slice:
            return self._get_slice(item)
        else:
            raise IndexError

    def save(self, episode: Episode) -> None:
        self._batch.append(episode)
        if len(self._batch) == self._batch_size:
            self.flush()

    def save_batch(self, episodes: List[Episode]) -> None:
        for e in episodes:
            self.save(e)

    def flush(self) -> None:
        for i, e in enumerate(self._batch):
            self._save_episode(episode=e)
            self._episodes += 1
        self._batch.clear()

    def _save_episode(self, episode: Episode) -> None:
        episode_data = self._init_groups(episode_no=self._episodes)
        limit = self._max_steps if self._max_steps and self._max_steps < episode.length else episode.length

        for k, v in episode.observations.items():
            episode_data['obs'].create_dataset(name=k, data=v[:limit], compression='gzip')

        for k, v in episode.actions.items():
            episode_data['action'].create_dataset(name=k, data=v[:limit], compression='gzip')

        episode_data.create_dataset(name='reward', data=episode.rewards[:limit], compression='gzip')
        episode_data.create_dataset(name='done', data=episode.done[:limit], compression='gzip')


    def _init_groups(self, episode_no: int):
        episode_group = self._file.create_group(str(episode_no))

        for key in self._keys:
            episode_group.create_group(key)

        return episode_group
