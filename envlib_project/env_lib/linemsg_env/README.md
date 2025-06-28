# Line Message Environment

A standard gym environment for multi-agent line message passing simulation.

## Overview

The Line Message Environment simulates a line of agents that must pass messages along the line while dealing with message loss and random events. This environment is designed for studying distributed communication protocols and multi-agent coordination.

## Features

- **Configurable number of agents**: Set the number of agents in the line
- **Adjustable observation neighborhood**: Control how many neighbors each agent can observe
- **Message loss simulation**: Realistic message passing with random loss events
- **Flexible episode length**: Configure maximum steps per episode
- **Multiple rendering modes**: Human-readable output and RGB array visualization
- **Standard gym interface**: Compatible with gymnasium and RL libraries

## Environment Parameters

- `num_agents` (int, default=10): Number of agents in the line
- `n_obs_neighbors` (int, default=1): Number of neighbors each agent can observe
- `max_iter` (int, default=50): Maximum number of steps per episode
- `render_mode` (str, optional): Rendering mode ('human', 'rgb_array', None)

## Action and Observation Spaces

### Action Space
- **Type**: `Discrete(2^num_agents)`
- **Description**: Each agent can take action 0 or 1, encoded as a single discrete action

### Observation Space
- **Type**: `Box(0, 3, shape=(num_agents, n_obs_neighbors * 2 + 1))`
- **Description**: Each agent observes its local state and neighboring states
- **Values**: 
  - 0: No message
  - 1: Has message
  - 2: Unknown/missing neighbor

## Reward Structure

- **Base reward**: 0.1 for each agent that has a message
- **First agent bonus**: 10x multiplier for the first agent
- **Total reward**: Sum of all agent rewards

## Message Passing Dynamics

1. **State update**: Agents update their state based on their own action and neighbors' states
2. **Message loss**: Random events can cause message loss with 80% probability
3. **Propagation**: Messages can propagate along the line based on agent actions

## Usage Examples

### Basic Usage

```python
from linemsg_env import LineMsgEnv

# Create environment
env = LineMsgEnv(num_agents=6, render_mode="human")

# Reset environment
obs, info = env.reset()

# Take actions
for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Different Configurations

```python
# Small line with local observations
env = LineMsgEnv(num_agents=3, n_obs_neighbors=0)

# Large line with extended neighborhood
env = LineMsgEnv(num_agents=15, n_obs_neighbors=2)

# Short episodes
env = LineMsgEnv(num_agents=8, max_iter=20)
```

## Installation

The environment is self-contained and requires:
- `numpy`
- `gymnasium`

## Testing

Run the test suite to verify the environment works correctly:

```bash
python test_env.py
```

## Examples

Run the example scripts to see different use cases:

```bash
python example.py
```

## Environment Registration

The environment is registered with gymnasium as `LineMsg-v0`:

```python
import gymnasium as gym
env = gym.make("LineMsg-v0", num_agents=5)
```

## Key Differences from Original

This new implementation provides:

1. **Standard gym interface**: Compatible with modern RL libraries
2. **Configurable parameters**: Easy to adjust environment properties
3. **Better documentation**: Clear API and usage examples
4. **Improved testing**: Comprehensive test suite
5. **Flexible rendering**: Multiple visualization options

## Research Applications

This environment is suitable for studying:
- Distributed communication protocols
- Multi-agent coordination
- Message routing algorithms
- Fault-tolerant systems
- Consensus protocols

## Contributing

Feel free to extend the environment with additional features such as:
- Different network topologies
- More complex message types
- Additional failure modes
- Performance metrics 