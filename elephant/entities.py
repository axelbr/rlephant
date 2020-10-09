from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np


@dataclass
class Transition:
    observation: Dict[str, np.ndarray]
    action: Dict[str, np.ndarray]
    reward: float
    done: bool

@dataclass
class Episode:

    observations: Dict[str, np.ndarray] = field(default_factory=lambda: {})
    actions: Dict[str, np.ndarray] = field(default_factory=lambda: {})
    rewards: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float))
    done: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool))

    def __getitem__(self, item) -> Iterable[Transition]:
        transitions = []
        start = item.start if item.start else 0
        stop = item.stop if item.stop else self.length
        step = item.step if item.step else 1
        for i in range(start, stop, step):
            obs = {}
            for k, v in self.observations.items():
                obs[k] = v[i]

            action = {}
            for k, v in self.actions.items():
                action[k] = v[i]

            reward = self.rewards[i]
            done = self.done[i]
            transitions.append(Transition(observation=obs,
                                          action=action,
                                          reward=reward,
                                          done=done))
        return transitions

    def __iter__(self):
       return iter(self[:])

    @property
    def length(self) -> int:
        return len(self.rewards)

    def append(self, transition: Transition):
        for k, v in transition.observation.items():
            if k in self.observations:
                self.observations[k] = np.vstack((self.observations[k], v))
            else:
                self.observations[k] = v

        for k, v in transition.action.items():
            if k in self.actions:
                self.actions[k] = np.vstack((self.actions[k], v))
            else:
                self.actions[k] = v

        self.rewards = np.append(self.rewards, transition.reward)
        self.done = np.append(self.done, transition.done)
