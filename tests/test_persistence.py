import os
import unittest
from glob import glob

from rlephant import ReplayStorage
from tests.util import make_episode


class TestReplayStorage(unittest.TestCase):

    def test_saving_single_episode(self):
        filename = 'episodes/single_episode.h5'
        episode = make_episode(length=200)
        storage = ReplayStorage(filename=filename)
        storage.save(episode)

        other = ReplayStorage(filename=filename)
        self.assertTrue(os.path.exists(filename))
        self.assertEqual(episode, other[0])

    def test_saving_single_episode_max_timesteps(self):
        filename = 'episodes/single_episode.h5'
        episode = make_episode(length=200)
        storage = ReplayStorage(filename=filename, max_steps=50)
        storage.save(episode)

        other = ReplayStorage(filename=filename)
        self.assertTrue(os.path.exists(filename))
        self.assertEqual(episode[:50], other[0])

    def test_saving_batched_episodes(self):
        filename = 'episodes/single_episode.h5'
        episodes = [make_episode(length=200) for i in range(10)]
        storage = ReplayStorage(filename=filename, batch_size=10)

        storage.save_batch(episodes[:9])
        self.assertEqual(0, len(storage))
        storage.save(episodes[9])
        self.assertEqual(10, len(storage))
        self.assertEqual(episodes[9], storage[-1])

    def tearDown(self) -> None:
        [os.remove(f) for f in glob('episodes/*')]
        os.rmdir('episodes')

    def setUp(self) -> None:
        os.mkdir('episodes')


if __name__ == '__main__':
    unittest.main()
