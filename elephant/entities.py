from dataclasses import dataclass, field
from typing import Dict, Iterable, Union, Iterator

import numpy as np


@dataclass
class Transition:
    observation: Dict[str, np.ndarray]
    action: Dict[str, np.ndarray]
    reward: float
    done: bool

    def __eq__(self, other):
        if not isinstance(other, Transition):
            return False

        if self.reward != other.reward or self.done != other.done:
            return False

        for k, v in self.observation.items():
            comp = other.observation[k] == v
            if not comp.all():
                return False

        for k, v in self.action.items():
            comp = other.action[k] == v
            if not comp.all():
                return False

        return True

@dataclass
class Episode:

    observations: Dict[str, np.ndarray] = field(default_factory=lambda: {})
    actions: Dict[str, np.ndarray] = field(default_factory=lambda: {})
    rewards: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float))
    done: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool))

    def _get_item(self, index: int) -> Transition:
        obs = {}
        for k, v in self.observations.items():
            obs[k] = v[index, :]

        action = {}
        for k, v in self.actions.items():
            action[k] = v[index, :]

        reward = self.rewards[index]
        done = self.done[index]
        return Transition(observation=obs,
                          action=action,
                          reward=reward,
                          done=done)

    def _get_slice(self, item) -> 'Episode':
        episode = Episode()
        for k, v in self.observations.items():
            episode.observations[k] = v[item]

        for k, v in self.actions.items():
            episode.actions[k] = v[item]

        episode.rewards = self.rewards[item]
        episode.done = self.done[item]
        return episode

    def __getitem__(self, item) -> Union[Transition, 'Episode']:
        if type(item) == int:
            return self._get_item(item)
        elif type(item) == slice:
            return self._get_slice(item)
        else:
            raise IndexError

    def __iter__(self) -> Iterator[Transition]:
       return iter([self._get_item(i) for i in range(self.length)])

    def __eq__(self, other) -> bool:
        if not isinstance(other, Episode):
            return False

        for k, v in self.observations.items():
            comp = other.observations[k] == v
            if not comp.all():
                return False

        for k, v in self.actions.items():
            comp = other.actions[k] == v
            if not comp.all():
                return False

        if not (other.rewards == self.rewards).all():
            return False

        if not (other.done == self.done).all():
            return False

        return True

    @property
    def length(self) -> int:
        return len(self.rewards)

    def append(self, transition: Transition) -> None:
        for k, v in transition.observation.items():
            v = np.expand_dims(v, axis=0)
            if k in self.observations:
                self.observations[k] = np.vstack((self.observations[k], v))
            else:
                self.observations[k] = v

        for k, v in transition.action.items():
            v = np.expand_dims(v, axis=0)
            if k in self.actions:
                self.actions[k] = np.vstack((self.actions[k], v))
            else:
                self.actions[k] = v

        self.rewards = np.append(self.rewards, transition.reward)
        self.done = np.append(self.done, transition.done)
