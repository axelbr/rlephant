from dataclasses import dataclass, field
from typing import Dict, Union, Iterator

import numpy as np

from .transitions import Transition, TransitionBatch


@dataclass
class Episode:
    """
    A single episode in an MDP. Observations and actions are stored as non-nested dictionaries. A key refers to a numpy
    array of arbitrary dimension. The first dimension corresponds to the time dimension.
    """

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
            if len(v.shape) == 2 and v.shape[1] == 1:
                action[k] = v[index, 0]
            else:
                action[k] = v[index, :]

        reward = self.rewards[index]
        done = self.done[index]
        return Transition(observation=obs,
                          action=action,
                          reward=reward,
                          done=done)

    def _get_slice(self, item) -> 'TransitionBatch':
        obs = {}
        actions = {}

        for k, v in self.observations.items():
            obs[k] = v[item]

        for k, v in self.actions.items():
            actions[k] = v[item]

        batch = TransitionBatch()
        batch.observations = obs
        batch.actions = actions
        batch.rewards = self.rewards[item]
        batch.done = self.done[item]
        return batch

    def __getitem__(self, item) -> Union[Transition, 'TransitionBatch']:
        """
        Access either a single transition at a particular timestep or slice and episode. If a single transition is
        requested, the caller will receive an object of type 'Transition'. If a slice is requested, the caller will
        receive a new episode, where the data (observations, actions, ...) is rearranged along the time axis, according
        to the given slice.
        :param item: either a single timestep (int) or a slice.
        :return: Either a single transition or a new episode with rearranged time-axis.
        """

        if type(item) == int:
            return self._get_item(item)
        elif type(item) == slice:
            return self._get_slice(item)
        else:
            raise IndexError

    def __iter__(self) -> Iterator[Transition]:
        """
        Iterator over all transitions.
        :return: Iterator of type Transition.
        """
        for i in range(self.length):
            yield self._get_item(i)

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

    def __len__(self):
        return self.length

    def append(self, transition: Union[Transition, TransitionBatch]) -> None:
        """
        Append a single transition to the episode. The dimensions of existing entries must match, otherwise, an exception
        will be thrown.
        :param transition: A single transition object.
        :return: None
        """
        if isinstance(transition, Transition):
            self._append_transition(transition)
        elif isinstance(transition, TransitionBatch):
            self._append_batch(transition)

    def _append_transition(self, transition: Transition):

        for k, v in transition.observation.items():
            if np.isscalar(v):
                v = np.array((v,))
            v = np.expand_dims(v, axis=0)
            if k in self.observations:
                self.observations[k] = np.vstack((self.observations[k], v))
            else:
                self.observations[k] = v

        for k, v in transition.action.items():
            if np.isscalar(v):
                v = np.array((v,))
            v = np.expand_dims(v, axis=0)
            if k in self.actions:
                self.actions[k] = np.vstack((self.actions[k], v))
            else:
                self.actions[k] = v

        self.rewards = np.append(self.rewards, transition.reward)
        self.done = np.append(self.done, transition.done)

    def _append_batch(self, batch: TransitionBatch):
        self.rewards = np.append(self.rewards, batch.rewards)
        self.done = np.append(self.done, batch.done)
        self._append_dict(self.observations, batch.observations)
        self._append_dict(self.actions, batch.actions)

    def _append_dict(self, member: Dict, item: Dict):
        for k, v in item.items():
            if k in member:
                member[k] = np.vstack((member[k], v))
            else:
                member[k] = v
