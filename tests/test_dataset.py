import os
import unittest
from glob import glob

from rlephant import Dataset
from tests.util import make_episode


class TestDataset(unittest.TestCase):

    def test_saving_single_episode(self):
        filename = 'episodes/single_episode.h5'
        episode = make_episode(length=200)
        storage = Dataset(filename=filename)
        storage.save(episode)

        other = Dataset(filename=filename)
        self.assertTrue(os.path.exists(filename))
        self.assertEqual(episode, other[0])

    def test_saving_single_episode_max_timesteps(self):
        filename = 'episodes/single_episode.h5'
        episode = make_episode(length=200)
        storage = Dataset(filename=filename, max_steps=50)
        storage.save(episode)

        other = Dataset(filename=filename)
        self.assertTrue(os.path.exists(filename))
        self.assertEqual(episode[:50], other[0][:])

    def test_saving_batched_episodes(self):
        filename = 'episodes/single_episode.h5'
        episodes = [make_episode(length=200) for i in range(10)]
        storage = Dataset(filename=filename, batch_size=10)

        storage.save_batch(episodes[:9])
        self.assertEqual(0, len(storage))
        storage.save(episodes[9])
        self.assertEqual(10, len(storage))
        self.assertEqual(episodes[9], storage[-1])

    def test_sample_sequences(self):
        filename = 'episodes/single_episode.h5'
        episodes = [make_episode(length=10) for i in range(4)]
        storage = Dataset(filename=filename)
        storage.save_batch(episodes)

        sequences = list(storage.sample_sequences(count=4, sequence_length=5))
        self.assertEqual(4, len(sequences))
        for seq in sequences:
            self.assertEqual(5, len(seq))

    def test_sample_sequence_too_long(self):
        filename = 'episodes/single_episode.h5'
        episodes = [make_episode(length=10) for i in range(4)]
        storage = Dataset(filename=filename)
        storage.save_batch(episodes)

        self.assertRaises(AssertionError, lambda: list(storage.sample_sequences(4, sequence_length=100)))

    def setUp(self) -> None:
        if os.path.exists('episodes'):
            [os.remove(f) for f in glob('episodes/*')]
            os.rmdir('episodes')
        os.mkdir('episodes')

    def tearDown(self) -> None:
        if os.path.exists('episodes'):
            [os.remove(f) for f in glob('episodes/*')]
            os.rmdir('episodes')


if __name__ == '__main__':
    unittest.main()
