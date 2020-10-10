import unittest

from rlephant import Episode
from tests.util import make_transition


class TestEpisode(unittest.TestCase):

    def test_single_append(self):
        transition = make_transition()
        episode = Episode()
        episode.append(transition)

        self.assertEqual(episode.length, 1, 'Episode should have length 1 after appending a single transition.')
        self.assertEqual(episode[0], transition)

    def test_multiple_appends(self):
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

    def test_slicing(self):
        episode = Episode()
        transitions = []
        for i in range(10):
            t = make_transition(i)
            episode.append(t)
            transitions.append(t)

        self.assertEqual(transitions[1], episode[1])

        self.assertEqual(transitions[:3], [t for t in episode[:3]])
        self.assertEqual(transitions[6:], [t for t in episode[6:]])
        self.assertEqual(transitions[3:6], [t for t in episode[3:6]])
        self.assertEqual(transitions[-5:-3], [t for t in episode[-5:-3]])
        self.assertEqual(transitions[::-1], [t for t in episode[::-1]])
        self.assertEqual(transitions[:-2], [t for t in episode[:-2]])
        self.assertEqual(transitions[-1:-2], [t for t in episode[-1:-2]])


if __name__ == '__main__':
    unittest.main()
