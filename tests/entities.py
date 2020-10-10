import unittest

import numpy as np

from elephant import Transition, Episode

def make_transition(value: int = None):
    if value is not None:
        return Transition(observation={'obs_a': np.array([[value, value], [value,value]]), 'obs_b': np.array([value])},
                   action={'action_a': np.array([value, -value]), 'action_b': np.array([value])},
                   reward=value,
                   done=True)
    else:
        return Transition(observation={'obs_a': np.random.uniform(-5, 5, size=(2,2)), 'obs_b': np.random.uniform(-5, 5, size=(2,))},
                          action={'action_a': np.random.uniform(-5, 5, size=(2,)), 'action_b': np.random.uniform(-5, 5, size=(2,))},
                          reward=np.random.uniform(-5, 5, size=(1,)),
                          done=np.random.choice([True, False]))


class TestEpisode(unittest.TestCase):

    def test_single_append(self):
        transition = make_transition()
        episode = Episode()
        episode.append(transition)

        self.assertEqual(episode.length, 1, 'Episode should have length 1 after appending a single transition.')
        self.assertEqual(episode.observations, transition.observation)
        self.assertEqual(episode.actions, transition.action)
        self.assertEqual(episode.rewards, transition.reward)
        self.assertEqual(episode.done, transition.done)

    def test_multiple_appends(self):
        transition1 = make_transition(value=1)
        transition2 = make_transition(value=2)
        episode = Episode()
        episode.append(transition1)
        episode.append(transition2)

        self.assertEqual(episode.length, 2)
        self.assertEqual(episode.observations['obs_a'].shape, (2, *transition1.observation['obs_a'].shape))
        self.assertEqual(episode.actions['action_a'].shape, (2, *transition1.action['action_a'].shape))
        self.assertEqual(episode.rewards.shape, (2,))
        self.assertEqual(episode.done.shape, (2,))

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
        self.assertEqual(transitions[:3], episode[:3])
        self.assertEqual(transitions[6:], episode[6:])
        self.assertEqual(transitions[3:6], episode[3:6])
        self.assertEqual(transitions[-5:-3], episode[-5:-3])
        self.assertEqual(transitions[::-1], episode[::-1])
        self.assertEqual(transitions[:-2], episode[:-2])
        self.assertEqual(transitions[-1:-2], episode[-1:-2])





if __name__ == '__main__':
    unittest.main()