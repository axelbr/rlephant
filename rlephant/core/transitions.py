from dataclasses import dataclass
from typing import Dict, Tuple

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
class TransitionBatch:
    observations: Dict[str, np.ndarray] = None
    actions: Dict[str, np.ndarray] = None
    rewards: np.ndarray = np.ndarray(shape=(1,), dtype=np.float)
    done: np.ndarray = np.ndarray(shape=(1,), dtype=np.bool)

    def _append_dict(self, item: Dict, member: Dict):
        for k, v in item.items():
            v = np.expand_dims(v, axis=0)
            member[k] = np.vstack((member[k], v))

    def append(self, transition: Transition):
        self.rewards = np.append(self.rewards, transition.reward)
        self.done = np.append(self.done, transition.done)
        self._append_dict(transition.observation, self.observations)
        self._append_dict(transition.action, self.actions)

    def __len__(self):
        return self.rewards.shape[0]

    def __eq__(self, other):
        if not isinstance(other, TransitionBatch):
            return False

        if not np.all(self.rewards == other.rewards) or not np.all(self.done == other.done):
            return False

        for k, v in self.observations.items():
            comp = other.observations[k] == v
            if not comp.all():
                return False

        for k, v in self.actions.items():
            comp = other.actions[k] == v
            if not comp.all():
                return False

        return True
