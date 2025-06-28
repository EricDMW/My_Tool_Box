# AJLATT: Advanced Joint Localization and Target Tracking Environment

A sophisticated multi-robot target tracking environment for reinforcement learning research, featuring advanced sensor models, dynamic obstacles, and realistic communication constraints. This environment simulates multiple robots tracking moving targets in 2D space with comprehensive sensor noise, obstacle avoidance, and multi-agent coordination capabilities.

## 🚀 Features

### Core Capabilities
- **Multi-robot coordination**: Support for multiple robots tracking multiple targets simultaneously
- **Advanced sensor models**: Range and bearing sensors with configurable noise parameters
- **Dynamic obstacle environments**: Both static and dynamic obstacle scenarios
- **Realistic communication constraints**: Limited communication range with information fusion
- **Comprehensive map system**: Extensive collection of pre-built maps and custom map creation
- **Gym-compatible interface**: Standard gym interface for seamless RL algorithm integration
- **Visualization tools**: Real-time rendering with customizable display options

### Advanced Features
- **Covariance Intersection (CI)**: Advanced information fusion for multi-robot state estimation
- **Target trajectory prediction**: Sophisticated target movement modeling
- **Collision avoidance**: Built-in safety mechanisms for robot-robot and robot-obstacle interactions
- **Process noise modeling**: Realistic motion uncertainty for both robots and targets
- **Configurable parameters**: Extensive parameter customization for research flexibility

## 📦 Installation

### Prerequisites
- Python 3.7+
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Matplotlib >= 3.3.0
- Gymnasium >= 0.21.0

### Installation Steps

```bash
# Clone the repository
git clone <repository-url>
cd ajlatt_env

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## 🎯 Quick Start

### Basic Usage

```python
import torch
import env_lib

# Create environment with default parameters
env = env_lib.ajlatt_env(map_name='obstacles05')
obs = env.reset()

# Run a simple episode
for step in range(1000):
    # Generate random actions for all robots (4 robots, 2 actions each)
    action = torch.rand(4, 2)  # [linear_vel, angular_vel] for each robot
    
    # Take environment step
    obs, reward, done, info = env.step(action)
    
    # Render the environment
    env.render()
    
    if done:
        obs = env.reset()
```

### Using Gym Interface

```python
import gym
import env_lib

# Register the environment
env_lib.register()

# Create environment
env = gym.make('TargetTracking-v0', map_name='obstacles05')
obs = env.reset()

# Run episode
for step in range(200):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    
    if done:
        obs = env.reset()
```

## 🔧 Environment Configuration

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_Robot` | int | 4 | Number of tracking robots |
| `num_targets` | int | 1 | Number of targets to track |
| `map_name` | str | 'obstacles05' | Map configuration name |
| `sensor_r_max` | float | 3.0 | Maximum sensor detection range |
| `commu_r_max` | float | 6.0 | Maximum communication range |
| `T_steps` | int | 120 | Maximum steps per episode |
| `sampling_period` | float | 0.5 | Time step duration (seconds) |

### Sensor Parameters

```python
# Sensor configuration
sensor_r_max = 3.0      # Maximum sensor range
sensor_r_min = 0.0      # Minimum sensor range
fov = 180.0             # Field of view (degrees)
sigma_p = 0.0           # Range measurement noise
sigma_r = 0.0           # Bearing measurement noise
sigma_th = 0.0          # Additional bearing noise
```

### Robot and Target Parameters

```python
# Robot configuration
robot_init_pose = [
    [7.5, 12.5, 0.0],   # [x, y, theta] for each robot
    [7.5, 10.5, 0.0],
    [10, 12.5, 0.0],
    [10.0, 10.5, 0.0]
]

# Target configuration
target_init_pose = [
    [10.5, 11.5, 0.0],  # [x, y, theta] for each target
    [13.5, 6.0, 0.0]
]
target_linear_velocity = 0.3  # Target movement speed
```

## 🗺️ Available Maps

### Static Maps
- `empty`: Open space without obstacles
- `empty05`: Small open space
- `emptySmall`: Compact open space
- `obstacles02-08`: Various obstacle configurations with increasing complexity
- `obstacles05`: Recommended default map with moderate complexity

### Dynamic Maps
- `dynamic_map`: Environment with moving obstacles
- `dynamic_map_2`: Alternative dynamic obstacle configuration

### Map Characteristics

| Map Name | Size | Obstacles | Complexity | Use Case |
|----------|------|-----------|------------|----------|
| `empty` | 50x50 | None | Low | Basic testing |
| `obstacles05` | 50x50 | Moderate | Medium | Standard training |
| `obstacles08` | 50x50 | Dense | High | Advanced scenarios |
| `dynamic_map` | 50x50 | Moving | High | Dynamic environments |

## 🎮 Action and Observation Spaces

### Action Space
Each robot has a 2-dimensional continuous action space:
- **Linear velocity**: [0.0, 2.0] m/s
- **Angular velocity**: [-π/4, π/4] rad/s

```python
action_space = spaces.Box(
    low=np.array([0.0, -pi/4]), 
    high=np.array([2.0, pi/4]), 
    shape=(2,), 
    dtype=np.float64
)
```

### Observation Space
The observation includes comprehensive state information:
- Robot states (position, velocity, heading, covariance)
- Target estimates (position, velocity, heading, covariance)
- Sensor measurements (range, bearing)
- Communication data from other robots
- Obstacle information

```python
observation_space = spaces.Box(
    low=-np.inf * ones(state_num), 
    high=np.inf * ones(state_num), 
    shape=(state_num,), 
    dtype=np.float64
)
```

## 🏆 Reward Function

The reward function encourages optimal tracking behavior:

```python
def get_reward(self, RT_obs):
    """
    Reward components:
    1. Tracking accuracy (minimize target estimation error)
    2. Sensor coverage (maintain target visibility)
    3. Collision avoidance (penalty for collisions)
    4. Energy efficiency (penalty for excessive movement)
    5. Communication effectiveness (reward for information sharing)
    """
```

### Reward Components
- **Tracking Error**: Negative reward proportional to target estimation uncertainty
- **Coverage Reward**: Positive reward for maintaining target visibility
- **Collision Penalty**: Large negative reward for robot-robot or robot-obstacle collisions
- **Movement Penalty**: Small negative reward for excessive velocity changes
- **Communication Reward**: Positive reward for effective information sharing

## 🔬 Advanced Features

### Covariance Intersection (CI)
The environment implements advanced information fusion using Covariance Intersection:

```python
def CI(self, s, y, infomation_form=0, no_weight=0, print_weight=0, xyfusion=0, plot=0):
    """
    Covariance Intersection for multi-robot state estimation.
    Combines estimates from multiple robots while maintaining consistency.
    """
```

### Target Movement Models
Customizable target movement patterns:

```python
class TargetMoving:
    def get_target_action(self, target_state, time_step):
        """
        Implement custom target movement logic.
        Returns: [linear_velocity, angular_velocity]
        """
        return action
```

### Dynamic Obstacle Support
Support for time-varying obstacle environments:

```python
class DynamicMap:
    def get_obstacles(self, time_step):
        """
        Return obstacles at given time step.
        Supports moving obstacles and changing environments.
        """
        return obstacles
```

## 📊 Examples

### Basic Training Loop

```python
import torch
import env_lib
import numpy as np

# Create environment
env = env_lib.ajlatt_env(
    map_name='obstacles05',
    num_Robot=4,
    num_targets=1
)

# Training loop
for episode in range(1000):
    obs = env.reset()
    episode_reward = 0
    
    for step in range(200):
        # Your policy here
        action = torch.rand(4, 2)  # Random policy for demonstration
        
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode}: Total Reward = {episode_reward:.2f}")
```

### Custom Target Controller

```python
from tool_box import Target_Moving

class CustomTargetController(Target_Moving):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_target_action(self, target_state, time_step):
        # Implement custom movement pattern
        # Example: Circular motion
        angular_vel = 0.2
        linear_vel = 0.3
        return np.array([linear_vel, angular_vel])
```

### Map Visualization

```python
import matplotlib.pyplot as plt
from maps import map_utils

# Load and visualize map
map_path = "env/maps/obstacles05.yaml"
grid_map = map_utils.GridMap(map_path=map_path)
grid_map.plot_map()
plt.show()
```

## 🧪 Testing and Validation

### Basic Test
```bash
python test_1_env.py
```

### Example Usage
```bash
python example.py
```

### Map Visualization
```bash
python plot_map.py
```

## 📁 Project Structure

```
ajlatt_env/
├── env/
│   ├── ajlatt_agent.py      # Main environment implementation
│   ├── maps/                # Map configurations and utilities
│   │   ├── *.yaml          # Map configuration files
│   │   ├── *.cfg           # Map data files
│   │   ├── map_utils.py    # Map utility functions
│   │   └── dynamic_map.py  # Dynamic obstacle support
│   ├── parameters/         # Configuration parameters
│   │   ├── ajlatt_para.py # Main parameter definitions
│   │   └── *.py           # Additional parameter files
│   └── tool_box/          # Utility functions
│       ├── agent_models.py # Robot and target models
│       ├── target_moving.py # Target movement controllers
│       └── util.py        # General utilities
├── docs/                  # Documentation
├── example.py            # Usage examples
├── test_1_env.py         # Basic testing
├── plot_map.py           # Map visualization
├── requirements.txt      # Dependencies
└── setup.py             # Package setup
```

## 🔧 Customization

### Creating Custom Maps

1. **Static Maps**: Create YAML configuration files
2. **Dynamic Maps**: Implement custom `DynamicMap` classes
3. **Target Controllers**: Extend `TargetMoving` class for custom movement patterns

### Parameter Tuning

Modify parameters in `env/parameters/ajlatt_para.py` or pass them directly:

```python
env = env_lib.ajlatt_env(
    map_name='obstacles05',
    sensor_r_max=5.0,
    commu_r_max=8.0,
    num_Robot=6,
    num_targets=2
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this environment in your research, please cite:

```bibtex
@article{ajlatt2024,
  title={AJLATT: Advanced Joint Localization and Target Tracking Environment for Multi-Robot Systems},
  author={Wang, Dongming},
  journal={arXiv preprint},
  year={2024}
}
```

## 📞 Contact

- **Author**: Dongming Wang
- **Email**: dongming.wang@email.ucr.edu
- **Project**: RL_LLM
- **Institution**: University of California, Riverside

## 🙏 Acknowledgments

- Built on top of Gymnasium for RL compatibility
- Uses PyTorch for efficient tensor operations
- Implements advanced sensor fusion algorithms
- Supports comprehensive multi-robot coordination scenarios

---

**Note**: This environment is designed for research in multi-robot systems, target tracking, and reinforcement learning. For production use, additional safety and validation measures may be required. 