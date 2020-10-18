import unittest

import numpy as np

from rlephant import Transition
from tests.util import make_transition, make_transition_batch


class TestTransition(unittest.TestCase):
    def test_eq(self):
        transition1 = make_transition(value=1, done=False)
        transition2 = make_transition(value=1, done=False)
        transition3 = make_transition(value=2, done=False)

        self.assertEqual(transition1, transition2)
        self.assertNotEqual(transition1, transition3)

    def test_append_to_batch(self):
        batch = make_transition_batch(size=10)

        obs = {}
        for k, v in batch.observations.items():
            obs[k] = v[0]

        actions = {}
        for k, v in batch.actions.items():
            actions[k] = v[0]

        transition = Transition(observation=obs, action=actions, reward=batch.rewards[0], done=batch.done[0])
        batch.append(transition)

        self.assertEqual(len(batch), 11)
        for k, v in batch.observations.items():
            comp = v[-1] == transition.observation[k]
            self.assertTrue(np.all(comp))
