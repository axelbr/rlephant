import os
from typing import Iterable

import h5py

from .entities import Episode


class Dataset:

    def __init__(self, filename: str, batch_size: int = None, max_steps: int = None):
        """
        Contstructor.
        :param filename: H5 file to open. If no such file exists, a new one will be created.
        :param batch_size: Number of episodes to keep in memory before writing out to file. If none provided, batch size
        is 1.
        :param max_steps: Maximum number of timesteps to save. If none provided, no limit is assumed.
        """
        self._batch_size = batch_size if batch_size else 1
        self._max_steps = max_steps
        self._batch = []
        self._filename = filename
        self._keys = ['obs', 'action']
        self._init_file(filename)

    def _init_file(self, filename: str):
        if not os.path.exists(filename):
            with h5py.File(self._filename, mode='a') as file:
                file.attrs['episode_count'] = 0

    def _get_item(self, file: h5py.File, index: int):
        index = index if index >= 0 else len(file) + index
        episode_data = file[str(index)]
        obs, action = {}, {}

        for k in episode_data['obs']:
            obs[k] = episode_data['obs'][k][:]

        for k in episode_data['action']:
            action[k] = episode_data['action'][k][:]

        rewards = episode_data['reward'][:]
        done = episode_data['done'][:]
        return Episode(observations=obs, actions=action, rewards=rewards, done=done)

    def _get_slice(self, file: h5py.File, item: slice):
        start = item.start if item.start else 0
        stop = item.stop if item.stop else len(file)
        step = item.step if item.step else 1

        if start < 0:
            start = len(file) + start

        if stop < 0:
            stop = len(file) + stop

        if step < 0:
            tmp = start
            start = stop - 1
            stop = tmp - 1

        episodes = []
        for i in range(start, stop, step):
            episode = self._get_item(file=file, index=i)
            episodes.append(episode)

        return episodes

    def __len__(self):
        """
        Number of episodes in this file.
        :return:
        """
        with h5py.File(self._filename, mode='r') as file:
            return file.attrs['episode_count']

    def __getitem__(self, item):
        """
        Either access a single episode or a slice of episodes in the file.
        :param item:
        :return:
        """
        with h5py.File(self._filename, mode='r') as file:
            if type(item) == int:
                return self._get_item(file=file, index=item)
            elif type(item) == slice:
                return self._get_slice(file=file, item=item)
            else:
                raise IndexError

    def __iter__(self):
        return iter(self[:])

    def save(self, episode: Episode) -> None:
        """
        Save a single episode. Note that it will be not persisted on disk, until the current batch is full or flush()
        is called.
        :param episode: A single episode.
        :return:
        """
        self._batch.append(episode)
        if len(self._batch) == self._batch_size:
            self.flush()

    def save_batch(self, episodes: Iterable[Episode]) -> None:
        """
        Save multiple episodes at once.
        :param episodes: An iterable of episodes.
        :return:
        """
        for e in episodes:
            self.save(e)

    def flush(self) -> None:
        """
        Write the current batch to disk and delete episodes from buffer.
        :return:
        """
        for e in self._batch:
            self._save_episode(episode=e)
        self._batch.clear()

    def _save_episode(self, episode: Episode) -> None:
        with h5py.File(self._filename, mode='a') as file:
            episode_data = self._init_groups(file=file, episode_no=file.attrs['episode_count'])
            limit = self._max_steps if self._max_steps and self._max_steps < episode.length else episode.length

            for k, v in episode.observations.items():
                episode_data['obs'].create_dataset(name=k, data=v[:limit], compression='gzip')

            for k, v in episode.actions.items():
                episode_data['action'].create_dataset(name=k, data=v[:limit], compression='gzip')

            episode_data.create_dataset(name='reward', data=episode.rewards[:limit], compression='gzip')
            episode_data.create_dataset(name='done', data=episode.done[:limit], compression='gzip')
            episode_data.attrs['length'] = limit
            file.attrs['episode_count'] += 1

    def _init_groups(self, file: h5py.File, episode_no: int):
        episode_group = file.create_group(str(episode_no))

        for key in self._keys:
            episode_group.create_group(key)

        return episode_group
