# Wireless Communication Environment

A standard gym environment for multi-agent wireless communication networks, rewritten as a clean, configurable gym environment.

## Overview

The Wireless Communication environment simulates a wireless communication network where multiple agents must coordinate to transmit packets through access points while avoiding interference. Each agent can choose to transmit to one of four neighboring access points or remain idle. The goal is to maximize successful packet transmissions while minimizing interference.

## Features

- **Configurable grid size**: Set the number of agents in x and y directions
- **Flexible network parameters**: Adjust packet arrival and transmission success probabilities
- **Configurable deadline horizon**: Set how far ahead agents can see packet deadlines
- **Adjustable observation neighborhood**: Control how much local information agents can observe
- **Standard gym interface**: Compatible with gymnasium/gym environments
- **Rendering support**: Both human and rgb_array rendering modes

## Installation

The environment is self-contained and requires the following dependencies:
- `gymnasium` (or `gym`)
- `numpy`

## Usage

### Basic Usage

```python
from envs.wireless_comm_new import WirelessCommEnv

# Create environment with default parameters
env = WirelessCommEnv(grid_x=6, grid_y=6)

# Reset environment
obs, info = env.reset()

# Take actions
for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_x` | int | 6 | Number of agents in x-direction |
| `grid_y` | int | 6 | Number of agents in y-direction |
| `ddl` | int | 2 | Deadline horizon for packet transmission |
| `packet_arrival_probability` | float | 0.8 | Probability of new packet arrival |
| `success_transmission_probability` | float | 0.8 | Probability of successful transmission |
| `n_obs_neighbors` | int | 1 | Number of neighbors to observe |
| `max_iter` | int | 50 | Maximum number of steps per episode |
| `render_mode` | str | None | Rendering mode ('human', 'rgb_array', None) |

### Action Space

`Discrete(5^n_agents)` where each agent has 5 actions:
- 0: No transmission (idle)
- 1: Transmit to upper-left access point
- 2: Transmit to lower-left access point  
- 3: Transmit to upper-right access point
- 4: Transmit to lower-right access point

### Observation Space

`Box(0, 2, shape=(n_agents, obs_dim))` where each agent gets:
- Local state information for the deadline horizon
- Information about neighboring agents (based on n_obs_neighbors)
- Packet status and deadline information

### Reward Function

The reward is based on successful packet transmissions:
- +1 for each successful packet transmission
- 0 for failed transmissions or idle actions
- Total reward is the sum across all agents

## Examples

### Different Configurations

```python
# Small grid for easier learning
env = WirelessCommEnv(grid_x=3, grid_y=3)

# Large grid for complex scenarios
env = WirelessCommEnv(grid_x=8, grid_y=8)

# High traffic scenario
env = WirelessCommEnv(grid_x=5, grid_y=5, packet_arrival_probability=0.9)

# Low reliability scenario
env = WirelessCommEnv(grid_x=4, grid_y=4, success_transmission_probability=0.5)

# Long deadline horizon
env = WirelessCommEnv(grid_x=6, grid_y=6, ddl=4)

# Large observation neighborhood
env = WirelessCommEnv(grid_x=5, grid_y=5, n_obs_neighbors=2)
```

### Training with RL Libraries

```python
# Compatible with stable-baselines3
from stable_baselines3 import PPO
from envs.wireless_comm_new import WirelessCommEnv

env = WirelessCommEnv(grid_x=4, grid_y=4)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Testing

Run the test script to verify the environment works correctly:

```bash
cd envs/wireless_comm_new
python test_env.py
```

This will run various tests including:
- Basic environment functionality
- Different configurations
- Network parameters
- Observation neighborhoods
- Visualization

## Registration

The environment is automatically registered with gymnasium when imported:

```python
import gymnasium as gym
from envs.wireless_comm_new import WirelessCommEnv

# Can be created using gym.make
env = gym.make("WirelessComm-v0", grid_x=4, grid_y=4)
```

## Differences from Original

This version differs from the original PettingZoo implementation in several ways:

1. **Standard gym interface**: Uses gymnasium/gym instead of PettingZoo
2. **Single agent perspective**: Treats all agents as a single agent with multi-dimensional actions
3. **Simplified observation space**: More compact observation representation
4. **Configurable parameters**: Easy to adjust network and game parameters
5. **Cleaner code structure**: Better organized and documented

## Network Dynamics

The environment simulates realistic wireless communication dynamics:

1. **Packet Arrivals**: New packets arrive probabilistically at each agent
2. **Access Point Selection**: Agents choose which access point to transmit to
3. **Interference**: Multiple agents transmitting to the same access point causes interference
4. **Deadlines**: Packets have deadlines and must be transmitted before expiring
5. **Success Probability**: Transmissions succeed probabilistically based on interference

## License

This environment is part of the DSDP project and follows the same license terms. 