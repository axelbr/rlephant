import math
import os
import sys

import numpy as np

from rlephant import ReplayStorage


def main():
    file = sys.argv[1]
    storage = ReplayStorage(filename=file)

    print(f'Collection: {file}')
    print(f'Episodes: {len(storage)}, Size: {os.stat(file).st_size / math.pow(1024, 2):.2f} MB')
    obs = ', '.join([f'{k}: {v.shape[1:]}' for k, v in storage[0].observations.items()])
    actions = ', '.join([f'{k}: {v.shape[1:]}' for k, v in storage[0].actions.items()])
    print(f'Observations: [{obs}]')
    print(f'Actions: [{actions}]')

    print('\n### Stats ###')
    episodes = storage[:]
    episodes_lengths = [len(e) for e in episodes]
    print(
        f'-) Episode Length: mean: {np.mean(episodes_lengths):.0f}, std: {np.std(episodes_lengths):.1f}, min: {np.min(episodes_lengths)}, max: {np.max(episodes_lengths)}')

    episodes_rewards = [sum(e.rewards) for e in episodes]
    print(
        f'-) Episode Rewards: mean: {np.mean(episodes_rewards):.2f}, std: {np.std(episodes_rewards):.2f}, min: {np.min(episodes_rewards)}, max: {np.max(episodes_rewards)}')


if __name__ == '__main__':
    main()
