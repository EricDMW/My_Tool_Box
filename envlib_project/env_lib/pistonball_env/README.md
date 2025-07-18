# Pistonball Environment

A comprehensive, configurable gym environment for multi-agent pistonball simulation with advanced features for reinforcement learning research.

## 🎯 Overview

The Pistonball environment simulates a physics-based cooperative game where multiple pistons work together to move a ball to the left wall. Each piston can move up or down to create a path for the ball, requiring coordinated control to achieve the objective efficiently.

### 🚀 Key Features

- **🔧 Configurable Multi-Agent Control**: Set the number of pistons from 1 to any reasonable number
- **⚡ Simultaneous Control**: Control all pistons at once with sequence-based actions
- **💰 Flexible Reward System**: Configurable movement penalties to encourage efficient control
- **🎮 Multiple Action Spaces**: Support for both continuous (-1 to 1) and discrete (0, 1, 2) actions
- **⚙️ Adjustable Physics**: Fine-tune ball mass, friction, elasticity, and other physics parameters
- **🎨 Human Control**: Manual policy for human interaction and debugging
- **📊 Standard Gym Interface**: Fully compatible with gymnasium/gym environments
- **🖼️ Rendering Support**: Both human and rgb_array rendering modes
- **🔢 Visual Piston Numbering**: Display piston order numbers for easy identification
- **🔍 Robust Validation**: Comprehensive action validation and error handling

## 📦 Installation

### Prerequisites

The environment is self-contained and requires the following dependencies:

```bash
pip install gymnasium pygame pymunk numpy
```

Or if using conda:

```bash
conda install -c conda-forge gymnasium pygame pymunk numpy
```

### Quick Start

```python
from pistonball_env import PistonballEnv

# Create environment with default parameters
env = PistonballEnv(n_pistons=20)

# Reset environment
obs, info = env.reset()

# Take actions - action is a sequence with dimension equal to number of pistons
for step in range(100):
    action = env.action_space.sample()  # Random action for all pistons
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## ⚙️ Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_pistons` | int | 20 | Number of piston agents (1 to any reasonable number) |
| `time_penalty` | float | -0.1 | Reward penalty per time step (encourages faster completion) |
| `continuous` | bool | True | Whether to use continuous actions (-1 to 1) or discrete (0, 1, 2) |
| `random_drop` | bool | True | Whether to randomly drop the ball at start |
| `random_rotate` | bool | True | Whether to randomly rotate the ball at start |
| `ball_mass` | float | 0.75 | Mass of the ball (affects physics behavior) |
| `ball_friction` | float | 0.3 | Friction coefficient of the ball (0.0 to 1.0) |
| `ball_elasticity` | float | 1.5 | Elasticity of the ball (bounciness, >0) |
| `max_cycles` | int | 125 | Maximum number of steps per episode |
| `render_mode` | str | None | Rendering mode ('human', 'rgb_array', None) |
| `movement_penalty` | float | 0.0 | Penalty for piston movement (negative value, 0.0 = no penalty) |
| `movement_penalty_threshold` | float | 0.01 | Minimum movement to trigger penalty |
| `terminated_condition` | bool | True | Whether to terminate when ball hits left wall |
| `leftmost_piston_reward` | float | 0.0 | Reward for leftmost piston when ball hits left wall |
| `termination_reward` | float | 0.5 | Reward for observable pistons when ball hits left wall |

## 🎮 Action Space

### Continuous Mode (Default)
```python
Box(-1, 1, shape=(n_pistons,), dtype=np.float32)
```

**Action Format**: Sequence of actions for all pistons
- **Shape**: `(n_pistons,)` - one action value per piston
- **Values**: 
  - `-1.0`: Move piston down
  - `0.0`: Keep piston stationary
  - `1.0`: Move piston up
  - Values in between: Proportional movement

**Example**:
```python
# For 5 pistons
action = np.array([1.0, -1.0, 0.0, 0.5, -0.5])
# Piston 1: Move up fully
# Piston 2: Move down fully  
# Piston 3: Stay still
# Piston 4: Move up half way
# Piston 5: Move down half way
```

### Discrete Mode
```python
Box(0, 2, shape=(n_pistons,), dtype=np.int32)
```

**Action Format**: Sequence of discrete actions for all pistons
- **Shape**: `(n_pistons,)` - one discrete action value per piston
- **Values**: 
  - `0`: Move piston down (-1 in continuous space)
  - `1`: Keep piston stationary (0 in continuous space)
  - `2`: Move piston up (1 in continuous space)

**Example**:
```python
# For 5 pistons
action = np.array([2, 0, 1, 2, 0])
# Piston 1: Move up (2)
# Piston 2: Move down (0)
# Piston 3: Stay still (1)
# Piston 4: Move up (2)
# Piston 5: Move down (0)
```

## 👁️ Observation Space

```python
Box(-inf, inf, shape=(n_pistons, 7), dtype=np.float32)
```

Each piston receives a 7-dimensional observation vector:

| Index | Description | Normalization |
|-------|-------------|---------------|
| 0 | Piston y-position | Normalized to [-1, 1] relative to center |
| 1 | Piston x-position | Normalized to [0, 1] across screen width |
| 2 | Ball x-position | Normalized to [0, 1] across screen width (0 if not observable) |
| 3 | Ball y-position | Normalized to [0, 1] across screen height (0 if not observable) |
| 4 | Ball x-velocity | Normalized by 15 (typical max velocity) (0 if not observable) |
| 5 | Ball y-velocity | Normalized by 8 (typical max velocity) (0 if not observable) |
| 6 | Ball angular velocity | Normalized by 8 (typical max angular velocity) (0 if not observable) |

**Note**: Ball information (indices 2-6) is only available to pistons within `kappa` hops of the ball's position. Pistons outside this range receive zeros for ball-related observations.

**Example Observation**:
```python
obs = np.array([
    [0.5, 0.1, 0.8, 0.6, 0.2, -0.1, 0.05],  # Piston 1 observation (can see ball)
    [0.2, 0.2, 0.8, 0.6, 0.2, -0.1, 0.05],  # Piston 2 observation (can see ball)
    [-0.1, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],   # Piston 3 observation (cannot see ball)
    # ... for all pistons
])
```

## 🏆 Reward Function

The reward is composed of multiple components:

### 1. Local Reward
Based on ball movement toward the goal (left wall):
```python
if prev_position > curr_position:
    local_reward = 0.5 * (prev_position - curr_position)  # Moving left (good)
else:
    local_reward = prev_position - curr_position  # Moving right (bad)
```

### 2. Time Penalty
Small negative reward per step to encourage faster completion:
```python
time_penalty = -0.1  # Configurable via time_penalty parameter
```

### 3. Movement Penalty (New Feature)
Penalty for piston movement to encourage efficient control:
```python
if movement > movement_penalty_threshold:
    penalty = movement_penalty * (movement_distance / pixels_per_position)
```

### 4. Termination Reward (New Feature)
When `terminated_condition=True` and the ball hits the left wall:
```python
if ball_hits_left_wall and piston_can_observe_ball:
    termination_reward = termination_reward_value  # Configurable (default: 0.5)
else:
    termination_reward = 0.0
```

### 5. Leftmost Piston Reward (New Feature)
When `terminated_condition=True` and the ball hits the left wall:
```python
if ball_hits_left_wall and piston_index == 0:  # Leftmost piston
    leftmost_piston_reward = leftmost_piston_reward_value
else:
    leftmost_piston_reward = 0.0
```

### 6. Total Reward
```python
total_reward = local_reward + time_penalty + movement_penalty_total + termination_reward + leftmost_piston_reward
```

## 💰 Movement Penalty System

The movement penalty feature allows you to penalize piston movement to encourage more efficient control strategies.

### Configuration Examples

```python
# No movement penalty (default behavior)
env = PistonballEnv(n_pistons=10, movement_penalty=0.0)

# Light movement penalty
env = PistonballEnv(
    n_pistons=10,
    movement_penalty=-0.05,  # Small penalty per unit of movement
    movement_penalty_threshold=0.01  # Only penalize if movement > threshold
)

# Heavy movement penalty
env = PistonballEnv(
    n_pistons=10,
    movement_penalty=-0.2,  # Large penalty per unit of movement
    movement_penalty_threshold=0.02  # Higher threshold
)
```

### How It Works

1. **Movement Detection**: Tracks piston position changes between steps
2. **Threshold Filtering**: Only applies penalty if movement exceeds threshold
3. **Proportional Penalty**: Penalty scales with movement distance
4. **Efficiency Encouragement**: Rewards agents that use minimal movements

### Use Cases

- **Efficiency Training**: Encourage agents to find optimal control strategies
- **Energy Conservation**: Simulate real-world energy constraints
- **Smooth Control**: Promote gradual, controlled movements
- **Research Experiments**: Study trade-offs between effectiveness and efficiency

## 🎯 Termination Condition

The environment supports configurable termination when the ball hits the left wall, with optional rewards for pistons that can observe the ball.

### Configuration

```python
# Default behavior - terminate when ball hits left wall
env = PistonballEnv(n_pistons=10, terminated_condition=True)

# Disable termination - ball can bounce off left wall
env = PistonballEnv(n_pistons=10, terminated_condition=False)
```

### Termination Reward System

When `terminated_condition=True` and the ball hits the left wall:

1. **Episode Termination**: The episode ends immediately
2. **Observable Pistons**: Pistons within `kappa` hops of the ball receive a reward specified by `termination_reward` (default: +0.5)
3. **Non-Observable Pistons**: Pistons outside the observation range receive no termination reward

### Example Scenarios

```python
# Scenario 1: Ball hits left wall, kappa=1, default termination reward
# Piston closest to ball and its immediate neighbors get +0.5 reward
env = PistonballEnv(n_pistons=10, terminated_condition=True, kappa=1)

# Scenario 2: Ball hits left wall, kappa=2, custom termination reward
# Piston closest to ball and pistons within 2 hops get +1.0 reward
env = PistonballEnv(n_pistons=10, terminated_condition=True, kappa=2, termination_reward=1.0)

# Scenario 3: No termination - ball bounces off left wall
env = PistonballEnv(n_pistons=10, terminated_condition=False)

# Scenario 4: Custom termination reward
env = PistonballEnv(n_pistons=10, terminated_condition=True, termination_reward=2.0)
```

### Use Cases

- **Goal-Oriented Training**: Clear success condition with immediate feedback
- **Cooperative Learning**: Rewards pistons that contribute to the goal
- **Research Flexibility**: Choose between termination and continuous play
- **Performance Evaluation**: Clear metrics for success/failure

## 🎯 Leftmost Piston Reward

The environment supports a special reward for the leftmost piston (piston index 0) when the ball hits the left wall.

### Configuration

```python
# Default behavior - no leftmost piston reward
env = PistonballEnv(n_pistons=10, leftmost_piston_reward=0.0)

# Add leftmost piston reward
env = PistonballEnv(n_pistons=10, leftmost_piston_reward=1.0)

# Negative reward for leftmost piston
env = PistonballEnv(n_pistons=10, leftmost_piston_reward=-0.5)
```

### Leftmost Piston Reward System

When `terminated_condition=True` and the ball hits the left wall:

1. **Leftmost Piston**: Piston index 0 receives an additional reward specified by `leftmost_piston_reward`
2. **Other Pistons**: No additional leftmost piston reward
3. **Combined Rewards**: Leftmost piston can receive both termination reward (specified by `termination_reward`) and leftmost piston reward

### Example Scenarios

```python
# Scenario 1: Leftmost piston gets extra reward (default termination reward)
env = PistonballEnv(n_pistons=10, terminated_condition=True, leftmost_piston_reward=1.0)
# When ball hits left wall:
# - Leftmost piston: +0.5 (termination) + 1.0 (leftmost) = +1.5 total
# - Other observable pistons: +0.5 (termination only)
# - Non-observable pistons: +0.0

# Scenario 2: Leftmost piston gets penalty (custom termination reward)
env = PistonballEnv(n_pistons=10, terminated_condition=True, leftmost_piston_reward=-0.5, termination_reward=1.0)
# When ball hits left wall:
# - Leftmost piston: +1.0 (termination) + (-0.5) (leftmost) = +0.5 total
# - Other observable pistons: +1.0 (termination only)
# - Non-observable pistons: +0.0

# Scenario 3: Custom termination and leftmost piston rewards
env = PistonballEnv(n_pistons=10, terminated_condition=True, leftmost_piston_reward=2.0, termination_reward=0.5)
# When ball hits left wall:
# - Leftmost piston: +0.5 (termination) + 2.0 (leftmost) = +2.5 total
# - Other observable pistons: +0.5 (termination only)
# - Non-observable pistons: +0.0
```

### Use Cases

- **Position-Based Rewards**: Reward pistons based on their spatial position
- **Responsibility Assignment**: Give extra responsibility to the leftmost piston
- **Research Experiments**: Study the effect of position-based rewards
- **Coordination Incentives**: Encourage the leftmost piston to take more active role

## 🎯 Usage Examples

### Basic Environment Setup

```python
from pistonball_env import PistonballEnv
import numpy as np

# Create environment
env = PistonballEnv(
    n_pistons=8,
    render_mode="human",  # For visualization
    continuous=True
)

# Reset and get initial observation
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
```

### Different Control Strategies

```python
# Strategy 1: All pistons move up
action = np.ones(env.n_pistons)
obs, reward, terminated, truncated, info = env.step(action)

# Strategy 2: All pistons move down
action = -np.ones(env.n_pistons)
obs, reward, terminated, truncated, info = env.step(action)

# Strategy 3: Stay still
action = np.zeros(env.n_pistons)
obs, reward, terminated, truncated, info = env.step(action)

# Strategy 4: Alternating pattern
action = np.array([1, -1, 1, -1, 1, -1, 1, -1])
obs, reward, terminated, truncated, info = env.step(action)

# Strategy 5: Wave pattern
action = np.array([1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5])
obs, reward, terminated, truncated, info = env.step(action)
```

### Training Loop Example

```python
import numpy as np
from pistonball_env import PistonballEnv

# Create environment with movement penalty
env = PistonballEnv(
    n_pistons=10,
    movement_penalty=-0.1,
    movement_penalty_threshold=0.01,
    render_mode=None  # No rendering for faster training
)

# Training loop
episode_rewards = []
for episode in range(100):
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(200):  # Max 200 steps per episode
        # Your agent's action selection here
        action = env.action_space.sample()  # Random for demonstration
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    episode_rewards.append(episode_reward)
    print(f"Episode {episode}: Total reward = {episode_reward:.2f}")

env.close()
```

### Physics Parameter Experiments

```python
# Heavy ball experiment
env_heavy = PistonballEnv(
    n_pistons=15,
    ball_mass=2.0,        # Heavy ball
    ball_friction=0.8,    # High friction
    ball_elasticity=0.5,  # Low bounce
    movement_penalty=-0.05
)

# Bouncy ball experiment
env_bouncy = PistonballEnv(
    n_pistons=15,
    ball_mass=0.5,        # Light ball
    ball_friction=0.1,    # Low friction
    ball_elasticity=2.0,  # High bounce
    movement_penalty=-0.1
)
```

### Discrete Action Space

```python
# Discrete action environment
env_discrete = PistonballEnv(
    n_pistons=5,
    continuous=False,  # Use discrete actions
    movement_penalty=-0.1
)

print(f"Discrete action space: {env_discrete.action_space}")
print(f"Action space shape: {env_discrete.action_space.shape}")

# Take discrete action - sequence of discrete actions for each piston
action = np.array([2, 0, 1, 2, 0])  # up, down, stay, up, down
obs, reward, terminated, truncated, info = env_discrete.step(action)

# Or sample random discrete actions
action = env_discrete.action_space.sample()  # Random discrete actions
obs, reward, terminated, truncated, info = env_discrete.step(action)
```

## 🎮 Manual Control

### Setup Manual Control

```python
from pistonball_env import PistonballEnv
from manual_policy import ManualPolicy

# Create environment with human rendering
env = PistonballEnv(
    n_pistons=6,
    render_mode="human",
    continuous=True
)

# Create manual policy
manual_policy = ManualPolicy(env)

# Manual control loop
obs, info = env.reset()
while True:
    action = manual_policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()
```

### Manual Control Keys

| Key | Action |
|-----|--------|
| **W** | Move selected piston up |
| **S** | Move selected piston down |
| **A** | Select previous piston |
| **D** | Select next piston |
| **ESC** | Exit the environment |
| **BACKSPACE** | Reset the environment |

## 🧪 Testing and Validation

### Run Basic Tests

```bash
# Run comprehensive test suite
python test_movement_penalty.py

# Run basic examples
python example.py

# Run interactive demo
python demo_enhanced_features.py
```

### Test Coverage

The test suite covers:
- ✅ Action validation and error handling
- ✅ Movement penalty functionality
- ✅ Different environment configurations
- ✅ Multi-piston control patterns
- ✅ Action clipping and type conversion
- ✅ Physics parameter variations
- ✅ Discrete vs continuous action spaces

### Custom Testing

```python
# Test action validation
env = PistonballEnv(n_pistons=5)
obs, info = env.reset()

# Correct action shape
action = np.ones(5)
obs, reward, terminated, truncated, info = env.step(action)

# Incorrect action shape (will raise error)
try:
    action = np.ones(3)  # Wrong shape
    obs, reward, terminated, truncated, info = env.step(action)
except ValueError as e:
    print(f"Correctly caught error: {e}")
```

## 🔧 Advanced Configuration

### Environment Registration

The environment is automatically registered with gymnasium:

```python
import gymnasium as gym
from pistonball_env import PistonballEnv

# Can be created using gym.make
env = gym.make("Pistonball-v0", n_pistons=10)
```

### Custom Reward Functions

You can extend the environment to add custom reward components:

```python
class CustomPistonballEnv(PistonballEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Add custom reward component
        custom_reward = self.calculate_custom_reward()
        reward += custom_reward
        
        return obs, reward, terminated, truncated, info
    
    def calculate_custom_reward(self):
        # Your custom reward logic here
        return 0.0
```

### Multi-Environment Training

```python
import numpy as np
from pistonball_env import PistonballEnv

# Create multiple environments with different configurations
envs = []
configs = [
    {"n_pistons": 5, "movement_penalty": -0.05},
    {"n_pistons": 10, "movement_penalty": -0.1},
    {"n_pistons": 15, "movement_penalty": -0.15}
]

for config in configs:
    env = PistonballEnv(**config, render_mode=None)
    envs.append(env)

# Train on multiple environments
for episode in range(100):
    for i, env in enumerate(envs):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Env {i}, Episode {episode}: Reward = {episode_reward:.2f}")

# Clean up
for env in envs:
    env.close()
```

## 📊 Performance Considerations

### Rendering Modes

- **`render_mode=None`**: Fastest, no visualization (recommended for training)
- **`render_mode="rgb_array"`**: Returns RGB array for logging/recording
- **`render_mode="human"`**: Interactive window (for debugging/visualization)

### Piston Numbering Feature

The environment includes a visual piston numbering feature that displays the order of each piston at the bottom during rendering:

- **Visual Indicators**: Each piston displays its index number (0, 1, 2, ...) at the bottom
- **White Text**: Numbers are rendered in white text for good visibility
- **Centered Positioning**: Numbers are centered at the bottom of each piston
- **Automatic Scaling**: Works with any number of pistons

**Example Usage**:
```python
# Create environment with visual rendering
env = PistonballEnv(n_pistons=8, render_mode="human")
obs, info = env.reset()

# The pistons will display numbers 0-7 at their bottoms
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Numbers will be visible in the rendered window
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

**Benefits**:
- **Easy Identification**: Quickly identify specific pistons during debugging
- **Action Mapping**: Correlate action indices with visual piston positions
- **Research Aid**: Useful for analyzing individual piston behavior
- **Teaching Tool**: Helpful for understanding multi-agent coordination

### Memory Usage

- **Small teams** (1-10 pistons): Low memory usage
- **Medium teams** (10-20 pistons): Moderate memory usage
- **Large teams** (20+ pistons): Higher memory usage

### Training Speed

- **No rendering**: ~1000+ steps/second
- **RGB array rendering**: ~100-500 steps/second
- **Human rendering**: ~20-50 steps/second

## 🔍 Troubleshooting

### Common Issues

1. **Action Shape Error**:
   ```python
   # Error: Action shape (3,) does not match expected shape (5,)
   # Solution: Ensure action has correct number of elements
   action = np.ones(env.n_pistons)  # Correct
   ```

2. **Rendering Issues**:
   ```python
   # If pygame rendering fails, try:
   env = PistonballEnv(render_mode="rgb_array")  # Alternative rendering
   ```

3. **Physics Instability**:
   ```python
   # For very large teams, adjust physics parameters:
   env = PistonballEnv(
       n_pistons=30,
       ball_mass=1.0,  # Stable mass
       ball_friction=0.5  # Moderate friction
   )
   ```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

env = PistonballEnv(n_pistons=5, render_mode="human")
```

## 🤝 Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python test_movement_penalty.py`
4. Make changes and test thoroughly

### Code Style

- Follow PEP 8 guidelines
- Add docstrings for all functions
- Include type hints where appropriate
- Write tests for new features

## 📚 References

### Related Work

- **PettingZoo Pistonball**: Original multi-agent implementation
- **Gymnasium**: Standard RL environment interface
- **Pymunk**: 2D physics engine
- **Pygame**: Graphics and input handling

### Research Applications

- Multi-agent reinforcement learning
- Cooperative control systems
- Resource allocation optimization
- Emergent behavior studies
- Efficiency vs effectiveness trade-offs

## 📄 License

This environment is part of the DSDP project and follows the same license terms.

## 🙏 Acknowledgments

- Original PettingZoo implementation
- Pymunk physics engine developers
- Gymnasium community
- Contributors and testers

---

**For questions, issues, or contributions, please refer to the project documentation or create an issue in the repository.** 