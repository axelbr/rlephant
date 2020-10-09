import collections
import os
from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple

import h5py
import numpy as np

@dataclass
class Episode:
    observations: List[Any] = field(default_factory=lambda: [])
    states: List[Dict[str, Any]] = field(default_factory=lambda: [])
    actions: List[Any] = field(default_factory=lambda: [])
    rewards: List[float] = field(default_factory=lambda: [])
    done: bool = False

    @property
    def length(self) -> int:
        return len(self.rewards)

    def append(self, step: Tuple[Any, Any, float, bool, Dict]):
        obs, action, reward, done, state = step
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.done = done
        self.states.append(state)

    def __iter__(self):
        dones = [False for _ in range(self.length - 1)] + [self.done]
        return iter([(o, a, r, d, s) for o, a, r, d, s in zip(self.observations, self.actions, self.rewards, dones, self.states)])



def unflatten(dictionary, sep='_'):
    resultDict = dict()
    for key, value in dictionary.iteritems():
        parts = key.split(sep)
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict

def flatten(dictionary, parent_key='', sep='_'):
    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class DataCollector:
    def __init__(self, directory: str, batch_size: int):
        self._batch_size = batch_size
        self._batch = []
        self._file = h5py.File(f'{directory}.hdf5')
        self._episodes = 0
        self._keys = ['obs', 'action', 'state']

    def save_episode(self, episode: Episode) -> None:
        episode_data = self.init_datasets(episode_no=self._episodes, episode=episode)
        t = 0
        for step in episode:
            self.save_step(episode_data=episode_data, step=t, data=step)
            t += 1
        self._episodes += 1

    def save_step(self, episode_data: h5py.Group, step: int, data: Tuple[Any, Any, float, bool, Dict]):
        o, a, r, d, s = data
        for k, v in flatten(o).items():
            episode_data['obs'][k][step] = v
        for k, v in flatten(a).items():
            episode_data['action'][k][step] = v
        for k, v in flatten(s).items():
            episode_data['state'][k][step] = v

        episode_data['reward'][step] = r
        episode_data['done'][step] = d

    def init_datasets(self, episode_no: int, episode: Episode):
        episode_group = self._file.create_group(str(episode_no))

        for key in self._keys:
            episode_group.create_group(key)

        episode_group.create_dataset(name='reward', shape=(episode.length, 1), dtype=np.float)
        episode_group.create_dataset(name='done', shape=(episode.length, 1), dtype=np.bool)

        def _init(name, dictionary):
            flattened = flatten(dictionary)
            for k, v in flattened.items():
                if hasattr(v, 'shape'):
                    episode_group[name].create_dataset(name=k, shape=(episode.length, *v.shape))
                else:
                    episode_group[name].create_dataset(name=k, shape=(episode.length, 1))

        _init('obs', episode.observations[0])
        _init('action', episode.actions[0])
        _init('state', episode.states[0])
        return episode_group




class DataReader:
    def __init__(self, filename: str):
        self._file = h5py.File(filename, 'r')

    def episode(self, index: int) -> Episode:
        episode_data = self._file[str(index)]
        obs, action, reward, done, state = episode_data['obs'], \
                                           episode_data['action'], \
                                           episode_data['reward'], \
                                           episode_data['done'], \
                                           episode_data['state']

        episode = Episode()