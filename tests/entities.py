import unittest

import numpy as np

from elephant import Transition, Episode


class TestEpisode(unittest.TestCase):

    def test_single_append(self):
        transition = Transition(observation={'obs_a': np.array([1.0, 1.0]), 'obs_b': np.array([2.0])},
                                action={'action_a': np.array([1.0, -1.0]), 'action_b': np.array([-3.0])},
                                reward=1.0,
                                done=True)
        episode = Episode()
        episode.append(transition)
        self.assertEqual(episode.length, 1, 'Episode should have length 1 after appending a single transition.')
        self.assertEqual(episode.observations, transition.observation)
        self.assertEqual(episode.actions, transition.action)
        self.assertEqual(episode.rewards, transition.reward)
        self.assertEqual(episode.done, transition.done)

    def test_multiple_appends(self):
        transition1 = Transition(observation={'obs_a': np.array([1.0, 1.0]), 'obs_b': np.array([2.0])},
                                action={'action_a': np.array([1.0, -1.0]), 'action_b': np.array([-3.0])},
                                reward=1.0,
                                done=True)
        transition2 = Transition(observation={'obs_a': np.array([3.0, 4.0]), 'obs_b': np.array([2.0])},
                                action={'action_a': np.array([7.0, -10.0]), 'action_b': np.array([-5.0])},
                                reward=0.0,
                                done=False)
        episode = Episode()
        episode.append(transition1)
        episode.append(transition2)

        self.assertEqual(episode.length, 2)
        self.assertEqual(episode.observations['obs_a'].shape, (2, *transition1.observation['obs_a'].shape))
        self.assertEqual(episode.actions['action_a'].shape, (2, *transition1.action['action_a'].shape))
        self.assertEqual(episode.rewards.shape, (2,))
        self.assertEqual(episode.done.shape, (2,))

        def test_iterator():
            transition1 = Transition(observation={'obs_a': np.array([1.0, 1.0]), 'obs_b': np.array([2.0])},
                                     action={'action_a': np.array([1.0, -1.0]), 'action_b': np.array([-3.0])},
                                     reward=1.0,
                                     done=True)








if __name__ == '__main__':
    unittest.main()