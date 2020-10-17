from dataclasses import dataclass, field
from typing import Dict, Union, Iterator, Tuple

import numpy as np


@dataclass
class Transition:
    """
    A single transition in a MDP, consisting of observations, actions, reward and done flag.
    It holds observations and actions as dictionaries.
    """
    observation: Dict[str, np.ndarray]
    action: Dict[str, np.ndarray]
    reward: float
    done: bool

    @staticmethod
    def from_tuple(transition: Tuple[Dict, Dict, float, bool]) -> 'Transition':
        observation, action, reward, done = transition
        return Transition(
            observation=observation,
            action=action,
            reward=reward,
            done=done
        )

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

    def __len__(self):
        return self.length

    def append(self, transition: Transition) -> None:
        """
        Append a single transition to the episode. The dimensions of existing entries must match, otherwise, an exception
        will be thrown.
        :param transition: A single transition object.
        :return: None
        """

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
