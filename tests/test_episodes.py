import unittest
from typing import Dict

import numpy as np

from rlephant import Transition, TransitionBatch
from rlephant.core import Episode
from tests.util import make_transition, make_transition_batch


class TestEpisode(unittest.TestCase):

    def test_single_append(self):
        transition = make_transition()
        episode = Episode()
        episode.append(transition)

        self.assertEqual(episode.length, 1, 'Episode should have length 1 after appending a single transition.')
        self.assertEqual(episode[0], transition)

    def test_batch_append(self):
        batch = make_transition_batch(size=100)
        episode = Episode()
        episode.append(batch)

        self.assertEqual(episode.length, len(batch))

        for k, v in batch.observations.items():
            comp = v == episode.observations[k]
            self.assertTrue(np.all(comp))

        for k, v in batch.actions.items():
            comp = v == episode.actions[k]
            self.assertTrue(np.all(comp))

        self.assertTrue(np.all(episode.done == batch.done))
        self.assertTrue(np.all(episode.rewards == batch.rewards))

    def test_append_multiple_batches(self):
        batch = make_transition_batch(size=10)
        episode = Episode()
        episode.append(batch)
        batch2 = make_transition_batch(size=40)
        episode.append(batch2)

        def _test_dict(actual, expected, from_idx, to_idx):
            for k, v in expected.items():
                comp = actual[k][from_idx:to_idx] == v
                self.assertTrue(np.all(comp))

        batch1_length = len(batch)
        batch2_length = len(batch2)
        self.assertEqual(episode.length, batch1_length + batch2_length)

        if isinstance(episode.observations, Dict):
            _test_dict(actual=episode.observations, expected=batch.observations, from_idx=0, to_idx=batch1_length)
            _test_dict(actual=episode.observations, expected=batch2.observations, from_idx=batch1_length,
                       to_idx=batch1_length + batch2_length)

        if isinstance(episode.actions, Dict):
            _test_dict(actual=episode.actions, expected=batch.actions, from_idx=0, to_idx=batch1_length)
            _test_dict(actual=episode.actions, expected=batch2.actions, from_idx=batch1_length,
                       to_idx=batch1_length + batch2_length)

    def test_append_multiple_transitions(self):
        transition1 = make_transition(value=1)
        transition2 = make_transition(value=2)
        episode = Episode()
        episode.append(transition1)
        episode.append(transition2)

        self.assertEqual(2, episode.length, 'Episode should have length 2 after appending two transitions.')
        self.assertEqual((2, *transition1.observation['obs_a'].shape), episode.observations['obs_a'].shape)
        self.assertEqual((2, *transition1.action['action_a'].shape), episode.actions['action_a'].shape)
        self.assertEqual((2,), episode.rewards.shape)
        self.assertEqual((2,), episode.done.shape)

    def test_iterator(self):
        episode = Episode()
        transitions = []
        for i in range(10):
            t = make_transition(i)
            episode.append(t)
            transitions.append(t)

        for i, t in enumerate(episode):
            self.assertEqual(t, transitions[i])

    def test_indexing(self):
        batch = make_transition_batch(size=100)
        episode = Episode()
        episode.append(batch)

        transition = episode[10]
        self.assertIsInstance(transition, Transition)
        for k in transition.observation:
            comp = batch.observations[k][10] == transition.observation[k]
            self.assertTrue(np.all(comp))

        for k in transition.action:
            comp = batch.actions[k][10] == transition.action[k]
            self.assertTrue(np.all(comp))

        self.assertEqual(batch.rewards[10], transition.reward)
        self.assertEqual(batch.done[10], transition.done)

    def test_slicing(self):
        batch = make_transition_batch(size=100)
        episode = Episode()
        episode.append(batch)

        accessor = slice(5, 10)
        transition = episode[accessor]
        self.assertIsInstance(transition, TransitionBatch)

        for k in transition.observations:
            comp = batch.observations[k][accessor] == transition.observations[k]
            self.assertTrue(np.all(comp))

        for k in transition.actions:
            comp = batch.actions[k][accessor] == transition.actions[k]
            self.assertTrue(np.all(comp))

        comp = (batch.rewards[accessor] == transition.rewards)
        self.assertTrue(np.all(comp))

        comp = (batch.done[accessor] == transition.done)
        self.assertTrue(np.all(comp))


if __name__ == '__main__':
    unittest.main()
