# RLephant

A simple tool for writing and reading episodes of reinforcement learning environments to and from disk.

## Features
- Convenient interface for constructing episodes and transitions in MDP's. Episodes can be accessed using
slicing along the time dimension.
- Efficient persistence on disk using the H5 file format.
- Only minimal dependencies.

## Installation

Install latest stable release from [pypi](https://pypi.org):
```bash
pip install rlepehant
```

For the latest version in the repository, you can directly install it from there:
```bash
pip install git+https://github.com/axelbr/elephant.git#egg=rlephant
```

## Usage

This simple example shows the basic usage of `rlephant`. More examples can be found in [examples](examples/).

```python
import rlephant
import gym

env = gym.make('CartPole-v0')
env.reset()

# Create an instance of ReplayStorage.
storage = rlephant.ReplayStorage('cartpole.h5')

# Create a new episode.
episode = rlephant.Episode()

action = env.action_space.sample()
obs, reward, done, info = env.step(action)

# Construct a new transition. Note that currently only
# dictionaries are supported for actions and observations.
transition = rlephant.Transition(
        observation={'some_obs': obs},
        action={'some_action': action},
        reward=reward,
        done=done)

# Append the transition to the episode...
episode.append(transition)

# ... and save it to disk.
storage.save(episode)

# Now you can access the episodes and transitions using slicing.
last_episode = storage[-1]
for transition in last_episode:
    print(transition)
```

## Tools

To print a summary of a collection, you can use the built in command line
tool `summary`. It will print information such as the number of episodes, episode 
stats etc. to the console.

*Usage:* `python -m rlephant.summary <path_to_collection>`