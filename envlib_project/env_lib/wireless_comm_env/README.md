# Wireless Communication Environment

A standard gym environment for multi-agent wireless communication simulation. This environment simulates a wireless communication network where agents must coordinate to transmit packets through access points while avoiding interference.

**Theoretical Foundation**: This implementation is based on the Networked MDP model for wireless communication networks with multiple access points, as described in the research literature. See [THEORETICAL_MODEL.md](THEORETICAL_MODEL.md) for detailed mapping to the theoretical framework.

## Features

- **Multi-agent coordination**: Multiple agents working in a grid-based environment
- **Realistic wireless simulation**: Packet arrival, transmission success probabilities, and interference modeling
- **Flexible rendering**: Support for real-time visualization and GIF creation
- **Standard gym interface**: Compatible with reinforcement learning frameworks
- **Customizable parameters**: Adjustable grid size, deadlines, probabilities, and more

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from wireless_comm_env import WirelessCommEnv

# Create environment
env = WirelessCommEnv(grid_x=4, grid_y=4, render_mode="rgb_array")
obs, info = env.reset()

# Run a simple episode
for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step + 1}: Reward = {reward}")
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_x` | 6 | Number of agents in x-direction |
| `grid_y` | 6 | Number of agents in y-direction |
| `ddl` | 2 | Deadline horizon for packet transmission |
| `packet_arrival_probability` | 0.8 | Probability of new packet arrival |
| `success_transmission_probability` | 0.8 | Probability of successful transmission |
| `n_obs_neighbors` | 1 | Number of neighbors to observe |
| `max_iter` | 50 | Maximum number of steps per episode |
| `render_mode` | None | Rendering mode ('human', 'rgb_array', None) |

## Action Space

Each agent has 5 possible actions:
- **0**: Idle (no transmission) - **Always legal**
- **1**: Transmit to Upper-Left access point - **Legal only if agent is not on top or left edge**
- **2**: Transmit to Lower-Left access point - **Legal only if agent is not on left edge**
- **3**: Transmit to Upper-Right access point - **Legal only if agent is not on top edge**
- **4**: Transmit to Lower-Right access point - **Legal only if agent is not on bottom or right edge**

### Illegal Action Handling

The environment automatically handles illegal actions by returning `None` for access point coordinates when an action would point to an invalid location. This means:
- Corner agents have only 2 legal actions (Idle + 1 transmission action)
- Edge agents have 3 legal actions (Idle + 2 transmission actions)
- Center agents have 5 legal actions (all actions)

## Observation Space

Each agent observes its local state including:
- Current packet status (deadline × local grid)
- Neighbor information (based on `n_obs_neighbors`)
- Total observation dimension: `ddl × ((n_obs_neighbors × 2 + 1)²)`

## Rendering System

The environment provides comprehensive rendering capabilities:

### Visual Elements
- **Blue cubes**: Represent agents (numbered 1-N, left-to-right, top-to-bottom)
- **Green circles**: Access points between agents
- **Pink arrows**: Successful transmissions (30% probability)
- **Black arrows**: General transmission attempts
- **Yellow boxes**: Current action labels
- **White boxes**: Agent state information

### Rendering Functions

#### Basic Rendering
```python
# Standard gym rendering
frame = env.render()  # Returns RGB array
```

#### Real-time Rendering
```python
# Show current state in real-time
env.render_realtime(show=True, save_path="current_frame.png")
```

#### GIF Creation
```python
# Collect frames during training
env.start_frame_collection()
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Automatically collects frames
    
env.stop_frame_collection()
env.save_gif("training_progress.gif", fps=3)
```

## Example Scripts

### 1. Simple Usage (`simple_usage_example.py`)
Demonstrates basic environment usage with different rendering options:
- Basic usage without rendering
- Standard rendering
- Real-time rendering
- GIF creation
- Custom parameters

### 2. Real-time Animation (`example_render_progress.py`)
Shows real-time animation during environment execution:
- Live visualization of agent movements
- Configurable frame delays
- Optional GIF saving

### 3. Separate Rendering Control (`example_separate_rendering.py`)
Demonstrates independent control of rendering modes:
- Real-time rendering only
- GIF saving only
- Both modes combined
- Custom configurations

## Usage Examples

### Basic Training Loop
```python
env = WirelessCommEnv(grid_x=4, grid_y=4, render_mode="rgb_array")
obs, info = env.reset()

total_reward = 0
for step in range(100):
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # Optional: render every N steps
    if step % 10 == 0:
        env.render_realtime(show=True)
    
    if terminated or truncated:
        obs, info = env.reset()
        total_reward = 0

env.close()
```

### Training with GIF Recording
```python
env = WirelessCommEnv(grid_x=4, grid_y=4, render_mode="rgb_array")
obs, info = env.reset()

# Start collecting frames
env.start_frame_collection()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Collects frame automatically
    
    if terminated or truncated:
        obs, info = env.reset()

# Save training progress as GIF
env.stop_frame_collection()
env.save_gif("training_progress.gif", fps=3)
env.close()
```

### Real-time Visualization
```python
env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="rgb_array")
obs, info = env.reset()

for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Show real-time progress
    env.render_realtime(show=True)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Performance Optimization

The environment has been optimized for:
- **Efficient rendering**: Backend-agnostic matplotlib usage
- **Memory management**: Automatic cleanup of rendering resources
- **Frame collection**: Optimized GIF creation with configurable FPS
- **Code structure**: Clean, maintainable codebase with proper error handling

## Troubleshooting

### Common Issues

1. **Matplotlib backend errors**: The environment uses 'Agg' backend for compatibility
2. **Memory usage**: Large GIFs can use significant memory; adjust FPS and frame count
3. **Display issues**: Real-time rendering requires a display; use `show=False` for headless systems

### Performance Tips

- Use `render_mode=None` for fastest training (no rendering overhead)
- Collect frames only when needed with `start_frame_collection()`/`stop_frame_collection()`
- Adjust `frame_delay` in real-time rendering for optimal viewing speed
- Use appropriate FPS values for GIF creation (2-5 FPS for training progress)

## Theoretical Model

This environment implements a **Networked MDP** model for wireless communication networks, capturing:

- **Multi-user coordination**: Multiple agents competing for shared access points
- **Packet dynamics**: Queues with deadlines and arrival probabilities  
- **Interference modeling**: Conflict detection when multiple agents transmit to same access point
- **Local interactions**: State transitions based on neighborhood structure
- **Binary state representation**: Packet status encoded as binary tuples

### Key Model Components

- **Users (Agents)**: N = {1, 2, ..., n} where n = grid_x × grid_y
- **Access Points**: Y = {y₁, y₂, ..., yₘ} at grid intersections
- **State Space**: Binary tuples sᵢ = (e₁, e₂, ..., e_{dᵢ}) ∈ {0,1}^{dᵢ}
- **Action Space**: Aᵢ = {null} ∪ Yᵢ (Idle + access point selection)
- **Reward Function**: rᵢ = 1 for successful transmission, 0 otherwise

For detailed theoretical mapping, see [THEORETICAL_MODEL.md](THEORETICAL_MODEL.md).

## Research Applications

This environment enables research in:
- **Multi-Agent Reinforcement Learning**: Test coordination algorithms
- **Network Optimization**: Study interference management strategies  
- **Protocol Design**: Evaluate wireless communication protocols
- **Resource Allocation**: Optimize access point utilization

## Contributing

This environment is part of a larger toolkit. For contributions or issues, please refer to the main project documentation.

## License

This project is part of the My_Tool_Box toolkit. See the main project for license information. 