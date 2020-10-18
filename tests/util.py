import numpy as np

from rlephant import Transition, Episode, TransitionBatch


def make_transition(value: int = None, done: bool = None):
    if value is not None:
        return Transition(observation={'obs_a': np.array([[value, value], [value, value]]), 'obs_b': np.array([value])},
                          action={'action_a': np.array([value, -value]), 'action_b': np.array([value])},
                          reward=value,
                          done=True)
    else:
        return Transition(
            observation={'obs_a': np.random.uniform(-5, 5, size=(20, 20)), 'obs_b': np.random.uniform(-5, 5, size=(2,))},
            action={'action_a': np.random.uniform(-5, 5, size=(2,)), 'action_b': np.random.uniform(-5, 5, size=(2,))},
            reward=np.random.uniform(-5, 5, size=(1,)),
            done=done if done is not None else np.random.choice([True, False]))


def make_transition_batch(size: int) -> TransitionBatch:
    batch = TransitionBatch()
    batch.observations = {
        'obs_a': np.random.uniform(0, 10, size=(size, 20, 10)),
        'obs_b': np.random.uniform(-2, 2, size=(size, 2))
    }
    batch.actions = {
        'action_a': np.random.uniform(0, 5, size=(size, 3))
    }
    batch.rewards = np.random.uniform(0, 2, size=(size,))
    batch.done = np.full(shape=(size,), fill_value=False)
    batch.done[-1] = True
    return batch

def make_episode(length: int):
    episode = Episode()
    for i in range(length-1):
        t = make_transition(done=False)
        episode.append(t)
    episode.append(make_transition(done=True))
    return episode
