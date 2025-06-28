# Pistonball Environment

A standard gym environment for the multi-agent pistonball game, rewritten as a clean, configurable gym environment.

## Overview

The Pistonball environment simulates a physics-based cooperative game where multiple pistons work together to move a ball to the left wall. Each piston can move up or down to create a path for the ball. The goal is to coordinate the pistons to successfully guide the ball to the left side of the screen.

## Features

- **Configurable number of agents**: Set the number of pistons from 1 to any reasonable number
- **Flexible physics parameters**: Adjust ball mass, friction, and elasticity
- **Continuous and discrete action spaces**: Support for both continuous (-1 to 1) and discrete (0, 1, 2) actions
- **Human control**: Manual policy for human interaction
- **Standard gym interface**: Compatible with gymnasium/gym environments
- **Rendering support**: Both human and rgb_array rendering modes

## Installation

The environment is self-contained and requires the following dependencies:
- `gymnasium` (or `gym`)
- `pygame`
- `pymunk`
- `numpy`

## Usage

### Basic Usage

```python
from envs.pistonball_new import PistonballEnv

# Create environment with default parameters
env = PistonballEnv(n_pistons=20)

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
| `n_pistons` | int | 20 | Number of piston agents |
| `time_penalty` | float | -0.1 | Reward penalty per time step |
| `continuous` | bool | True | Whether to use continuous actions |
| `random_drop` | bool | True | Whether to randomly drop the ball |
| `random_rotate` | bool | True | Whether to randomly rotate the ball |
| `ball_mass` | float | 0.75 | Mass of the ball |
| `ball_friction` | float | 0.3 | Friction coefficient of the ball |
| `ball_elasticity` | float | 1.5 | Elasticity of the ball |
| `max_cycles` | int | 125 | Maximum number of steps per episode |
| `render_mode` | str | None | Rendering mode ('human', 'rgb_array', None) |

### Action Space

- **Continuous mode**: `Box(-1, 1, shape=(n_pistons,))` - Each value controls one piston
- **Discrete mode**: `Discrete(3^n_pistons)` - Each piston has 3 actions (down, stay, up)

### Observation Space

`Box(-inf, inf, shape=(n_pistons, 6))` where each piston gets:
- Piston position (normalized)
- Ball x-position (normalized)
- Ball y-position (normalized)
- Ball x-velocity (normalized)
- Ball y-velocity (normalized)
- Ball angular velocity (normalized)

### Reward Function

The reward is based on:
- Local reward: How much the ball moved left when near pistons
- Time penalty: Small negative reward per step
- Global reward: Overall ball movement

## Manual Control

You can control the environment manually using the `ManualPolicy`:

```python
from envs.pistonball_new import PistonballEnv
from envs.pistonball_new.manual_policy import ManualPolicy

env = PistonballEnv(n_pistons=10, render_mode="human")
manual_policy = ManualPolicy(env)

obs, info = env.reset()

while True:
    action = manual_policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()
```

### Manual Control Keys

- **W/S**: Move selected piston up/down
- **A/D**: Select previous/next piston
- **ESC**: Exit
- **BACKSPACE**: Reset environment

## Examples

### Different Configurations

```python
# Few pistons for easier learning
env = PistonballEnv(n_pistons=5, continuous=True)

# Heavy ball for different physics
env = PistonballEnv(n_pistons=15, ball_mass=2.0, ball_friction=0.8)

# Bouncy ball
env = PistonballEnv(n_pistons=10, ball_elasticity=2.0)

# Discrete actions
env = PistonballEnv(n_pistons=8, continuous=False)
```

### Training with RL Libraries

```python
# Compatible with stable-baselines3
from stable_baselines3 import PPO
from envs.pistonball_new import PistonballEnv

env = PistonballEnv(n_pistons=10)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Testing

Run the test script to verify the environment works correctly:

```bash
cd envs/pistonball_new
python test_env.py
```

This will run various tests including:
- Basic environment functionality
- Different configurations
- Physics parameters
- Manual control (optional)

## Registration

The environment is automatically registered with gymnasium when imported:

```python
import gymnasium as gym
from envs.pistonball_new import PistonballEnv

# Can be created using gym.make
env = gym.make("Pistonball-v0", n_pistons=10)
```

## Differences from Original

This version differs from the original PettingZoo implementation in several ways:

1. **Standard gym interface**: Uses gymnasium/gym instead of PettingZoo
2. **Single agent perspective**: Treats all pistons as a single agent with multi-dimensional actions
3. **Simplified observation space**: More compact observation representation
4. **Configurable parameters**: Easy to adjust physics and game parameters
5. **Cleaner code structure**: Better organized and documented

## License

This environment is part of the DSDP project and follows the same license terms. 