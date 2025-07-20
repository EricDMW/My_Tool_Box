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

## Detailed Observation and Info Structure

### Observation Array
- For each agent, the observation is a flattened array of shape `(ddl * ((2 * n_obs_neighbors + 1) ** 2),)`.
- `ddl` is the deadline horizon for packet transmission.
- `n_obs_neighbors` determines the local neighborhood size (e.g., 1 means a 3x3 grid centered on the agent).
- The observation encodes the packet status for each cell in the agent's local neighborhood for each deadline slot.
- Values in the observation:
  - `0`: No packet
  - `1`: Packet present
  - `2`: Out-of-bounds (padding)

#### Detailed Structure of the Observation Array
Each digit in the observation array corresponds to a specific deadline slot and a specific cell in the agent's local neighborhood grid. The array is constructed by flattening the 3D array with axes:
- Deadline slot (from 0 to ddl-1)
- Local grid row (from -n_obs_neighbors to +n_obs_neighbors, relative to agent)
- Local grid column (from -n_obs_neighbors to +n_obs_neighbors, relative to agent)

The flattening order is: for each deadline slot, scan the local grid row by row (top to bottom, then left to right within each row).

##### Example: ddl=2, n_obs_neighbors=1
- The local grid is 3x3 (centered on the agent), and there are 2 deadline slots.
- The observation array has 18 elements: `obs[0]` to `obs[17]`.
- The mapping is as follows:

| Obs Index | Deadline | d_row | d_col | Meaning (relative to agent) |
|-----------|----------|-------|-------|-----------------------------|
| 0         | 0        | -1    | -1    | Top-left neighbor, earliest deadline |
| 1         | 0        | -1    |  0    | Top-center neighbor, earliest deadline |
| 2         | 0        | -1    | +1    | Top-right neighbor, earliest deadline |
| 3         | 0        |  0    | -1    | Center-left neighbor, earliest deadline |
| 4         | 0        |  0    |  0    | Agent itself, earliest deadline |
| 5         | 0        |  0    | +1    | Center-right neighbor, earliest deadline |
| 6         | 0        | +1    | -1    | Bottom-left neighbor, earliest deadline |
| 7         | 0        | +1    |  0    | Bottom-center neighbor, earliest deadline |
| 8         | 0        | +1    | +1    | Bottom-right neighbor, earliest deadline |
| 9         | 1        | -1    | -1    | Top-left neighbor, next deadline |
| 10        | 1        | -1    |  0    | Top-center neighbor, next deadline |
| 11        | 1        | -1    | +1    | Top-right neighbor, next deadline |
| 12        | 1        |  0    | -1    | Center-left neighbor, next deadline |
| 13        | 1        |  0    |  0    | Agent itself, next deadline |
| 14        | 1        |  0    | +1    | Center-right neighbor, next deadline |
| 15        | 1        | +1    | -1    | Bottom-left neighbor, next deadline |
| 16        | 1        | +1    |  0    | Bottom-center neighbor, next deadline |
| 17        | 1        | +1    | +1    | Bottom-right neighbor, next deadline |

- For each deadline slot, the 3x3 grid is flattened row by row, then the next deadline slot is appended.
- For larger `n_obs_neighbors` or `ddl`, the pattern generalizes accordingly.

### Agent and Access Point Layout
- Agents are arranged in a 2D grid of size `grid_x` by `grid_y`.
- Each agent is indexed in row-major order (left-to-right, top-to-bottom), starting from 0.
- Access points (APs) are located between agents, forming a grid of size `(grid_x-1)` by `(grid_y-1)`.
- Each AP is indexed in row-major order (top-to-bottom, left-to-right), starting from 1.
- For an AP at coordinates `(ap_x, ap_y)`, its index is: `ap_id = ap_y * (grid_x - 1) + ap_x + 1`.

## GPU and Torch Support

This environment supports both CPU and GPU computation using PyTorch tensors. You can specify the device with the `device` parameter (e.g., `'cpu'` or `'cuda'`).

- **All state, actions, and random sampling are performed on the specified device.**
- **Data is only moved to CPU (with `.cpu().numpy()`) when returning observations, info, or for rendering/saving frames.**
- **The environment is compatible with both NumPy and PyTorch workflows.**

### Example: Running on GPU
```python
import torch
from wireless_comm_env import WirelessCommEnv

device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = WirelessCommEnv(grid_x=6, grid_y=6, device=device)
obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, total_reward, terminated, truncated, info = env.step(action)
    print('Agent 0 local_obs:', info['local_obs'][0])
    if terminated or truncated:
        break
```

## Info Dictionary (Returned by `step`)
The `info` dictionary returned by `env.step(action)` contains additional information about the environment state. Its contents depend on the `debug_info` parameter:

- If `debug_info=False` (default):
  - `info['local_rewards']`: A numpy array of shape `(n_agents,)` with the reward for each agent in the current step.
  - `info['local_obs']`: A list of numpy arrays, where `info['local_obs'][i]` is a 1D array of length `ddl` containing the agent's own state (packet status for all deadlines) for agent `i`.

- If `debug_info=True`:
  - `info['local_rewards']`: As above.
  - `info['neighbors']`: A list of lists, where `info['neighbors'][i]` contains the indices of the four direct neighbors (up, down, left, right) of agent `i`.
  - `info['access_points']`: A list of lists, where `info['access_points'][i]` contains the indices (order) of all access points adjacent to agent `i` (i.e., all APs the agent could possibly access, not just the one chosen by its action).
  - `info['local_obs']`: As above.

### Example Usage
```python
obs, total_reward, terminated, truncated, info = env.step(action)

# For agent i:
local_reward = info['local_rewards'][i]
local_obs = info['local_obs'][i]  # 1D array of length ddl
neighbors = info['neighbors'][i]  # Only if debug_info=True
possible_ap_indices = info['access_points'][i]  # Only if debug_info=True
```

## Best Practices for Efficiency
- **Keep all computation on the GPU** by setting `device='cuda'` if available. Only move data to CPU for logging, rendering, or numpy compatibility.
- **Batch multiple environments** for large-scale RL to maximize GPU utilization.
- **Use efficient data types** (e.g., `torch.uint8` for binary state) to save memory and bandwidth.
- **Minimize Python loops** and use vectorized torch operations wherever possible.
- **Profile your code** to identify bottlenecks, especially in the `step` and `render` methods.
- **Avoid unnecessary data transfers** between CPU and GPU.
- **Downsample or skip frames** when saving GIFs to reduce memory and I/O overhead.

## Troubleshooting GPU/CPU Performance
- For small environments, CPU may be faster due to lower overhead. GPU shines with large, batched workloads.
- Data transfer between CPU and GPU is slow; keep as much computation as possible on the GPU.
- Use `torch.cuda.synchronize()` and profilers to measure true GPU time.
- If you see only the first/last frame in GIFs, check that all frames are the same size and type.
- Printing/logging in tight loops can slow down training.

## Data Types and Conversion
- **State, actions, and rewards are torch tensors on the specified device.**
- **Observations and info dicts are converted to numpy arrays (on CPU) before being returned to the user.**
- **Rendering always uses CPU numpy arrays.**

## Example: Extracting Local Observations
```python
obs, total_reward, terminated, truncated, info = env.step(action)
print('Agent 0 local_obs:', info['local_obs'][0])  # e.g., [0. 1.] for ddl=2
```

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