# Kuramoto Oscillator Environment Implementation

A comprehensive implementation of the Kuramoto oscillator environment with support for both dynamic and constant coupling modes, available in both NumPy and PyTorch versions.

## Overview

The Kuramoto model describes the dynamics of coupled oscillators:
```
dθᵢ/dt = ωᵢ + (K/N) * Σⱼ sin(θⱼ - θᵢ)
```

This implementation provides:
- **Dynamic Coupling**: Coupling strengths controlled by actions
- **Constant Coupling**: Fixed coupling matrix provided at initialization
- **Multiple Reward Types**: order_parameter, phase_coherence, frequency_synchronization
- **Custom Network Topologies**: fully connected, ring, star, clusters, random
- **GPU Support**: PyTorch version with CUDA acceleration
- **Multi-Agent Support**: PyTorch version supports parallel agents

## Features

### Coupling Modes

#### Dynamic Coupling (Default)
- **Action Space**: `[control_inputs, coupling_strengths]`
- **Observation Space**: `[phases, natural_frequencies, coupling_strengths, control_inputs]`
- **Control**: Both control inputs and coupling strengths are learned

#### Constant Coupling
- **Action Space**: `[control_inputs]` only
- **Observation Space**: `[phases, natural_frequencies, control_inputs]`
- **Control**: Only control inputs are learned, coupling matrix is fixed

### Network Topologies

- **Fully Connected**: All oscillators connected to each other
- **Ring**: Each oscillator connected to neighbors
- **Star**: Central oscillator connected to all others
- **Two Clusters**: Two disconnected fully connected clusters
- **Random**: Random connections with specified probability

### Reward Types

- **order_parameter**: Based on phase synchronization (0-1)
- **phase_coherence**: Alternative synchronization measure
- **frequency_synchronization**: Based on frequency deviation from target
- **combined**: Combination of multiple measures

## Installation

1. Install dependencies:
```bash
pip install -r requirements_kuramoto.txt
```

2. Ensure the `kos_env` package is available in your Python path.

## Quick Start

### Basic Usage

```python
from kuramoto_env_implementation import KuramotoEnvironmentFactory, TopologyGenerator, CouplingMatrixGenerator

# Create a simple NumPy environment
env = KuramotoEnvironmentFactory.create_numpy_env(
    n_oscillators=6,
    coupling_mode="dynamic",
    reward_type="frequency_synchronization",
    target_frequency=2.0
)

# Reset and run
obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
env.close()
```

### Constant Coupling Mode

```python
# Create constant coupling matrix
constant_matrix = CouplingMatrixGenerator.uniform(n_oscillators=6, strength=2.0)

# Create environment with constant coupling
env = KuramotoEnvironmentFactory.create_numpy_env(
    n_oscillators=6,
    coupling_mode="constant",
    constant_coupling_matrix=constant_matrix,
    reward_type="frequency_synchronization"
)
```

### Custom Topology

```python
# Create custom topology
adj_matrix = TopologyGenerator.two_clusters(n_oscillators=6)

# Create environment with custom topology
env = KuramotoEnvironmentFactory.create_numpy_env(
    n_oscillators=6,
    adj_matrix=adj_matrix,
    coupling_mode="dynamic"
)
```

### PyTorch Multi-Agent

```python
# Create PyTorch multi-agent environment
env = KuramotoEnvironmentFactory.create_pytorch_env(
    n_oscillators=6,
    n_agents=4,
    coupling_mode="constant",
    device="auto"  # Automatically use GPU if available
)
```

## Advanced Usage

### Environment Factory

The `KuramotoEnvironmentFactory` provides convenient methods for creating environments:

```python
# NumPy environment
env_numpy = KuramotoEnvironmentFactory.create_numpy_env(
    n_oscillators=8,
    coupling_mode="dynamic",
    reward_type="order_parameter",
    max_steps=500,
    dt=0.01
)

# PyTorch environment
env_pytorch = KuramotoEnvironmentFactory.create_pytorch_env(
    n_oscillators=8,
    n_agents=2,
    coupling_mode="constant",
    device="cuda",
    integration_method="rk4"
)
```

### Topology Generator

Generate different network topologies:

```python
# Different topologies
fully_connected = TopologyGenerator.fully_connected(6)
ring = TopologyGenerator.ring(6)
star = TopologyGenerator.star(6)
two_clusters = TopologyGenerator.two_clusters(6)
random = TopologyGenerator.random(6, connection_prob=0.3)
```

### Coupling Matrix Generator

Generate different coupling matrices:

```python
# Different coupling matrices
uniform = CouplingMatrixGenerator.uniform(6, strength=2.0)
distance_based = CouplingMatrixGenerator.distance_based(6, base_strength=2.0, decay=0.5)
hierarchical = CouplingMatrixGenerator.hierarchical(6, base_strength=2.0)
```

### Environment Testing

Test and evaluate environment performance:

```python
from kuramoto_env_implementation import EnvironmentTester

# Create environment
env = KuramotoEnvironmentFactory.create_numpy_env(n_oscillators=6)

# Test with random policy
tester = EnvironmentTester(env, n_episodes=10, max_steps=200)
results = tester.run_random_policy()

# Plot results
fig = tester.plot_results(results, "My Environment Performance")
plt.show()

# Print statistics
print(f"Average Reward: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}")
print(f"Average Order Parameter: {results['avg_order_param']:.3f} ± {results['std_order_param']:.3f}")
```

## Demos

The implementation includes several demonstration functions:

### Dynamic vs Constant Coupling Comparison

```python
from kuramoto_env_implementation import demo_dynamic_vs_constant

# Compare dynamic and constant coupling modes
demo_dynamic_vs_constant()
```

### PyTorch Multi-Agent Demo

```python
from kuramoto_env_implementation import demo_pytorch_multi_agent

# Demonstrate multi-agent environment
demo_pytorch_multi_agent()
```

### Different Topologies Demo

```python
from kuramoto_env_implementation import demo_different_topologies

# Compare different network topologies
demo_different_topologies()
```

## Environment Parameters

### Common Parameters

- `n_oscillators`: Number of oscillators in the system
- `coupling_mode`: "dynamic" or "constant"
- `reward_type`: "order_parameter", "phase_coherence", "frequency_synchronization", "combined"
- `target_frequency`: Target frequency for frequency_synchronization reward
- `max_steps`: Maximum steps per episode
- `dt`: Time step for integration

### NumPy-Specific Parameters

- `noise_std`: Standard deviation of noise added to dynamics
- `render_mode`: "human" or "rgb_array"

### PyTorch-Specific Parameters

- `n_agents`: Number of parallel agents
- `device`: "cpu", "cuda", or "auto"
- `integration_method`: "euler" or "rk4"

## Action and Observation Spaces

### Dynamic Coupling Mode

**Action Space**: `Box(low=[control_lows, coupling_lows], high=[control_highs, coupling_highs])`
- Control inputs: `n_oscillators` values in `[-1.0, 1.0]`
- Coupling strengths: `n_couplings` values in `[0.0, 5.0]`

**Observation Space**: `Box(low=-inf, high=inf, shape=(n_oscillators*3 + n_couplings,))`
- Phases: `n_oscillators` values
- Natural frequencies: `n_oscillators` values
- Coupling strengths: `n_couplings` values
- Control inputs: `n_oscillators` values

### Constant Coupling Mode

**Action Space**: `Box(low=control_lows, high=control_highs)`
- Control inputs: `n_oscillators` values in `[-1.0, 1.0]`

**Observation Space**: `Box(low=-inf, high=inf, shape=(n_oscillators*3,))`
- Phases: `n_oscillators` values
- Natural frequencies: `n_oscillators` values
- Control inputs: `n_oscillators` values

## Performance Tips

1. **Use GPU**: PyTorch version is significantly faster on GPU
2. **Choose Coupling Mode**: Constant coupling has smaller action space for easier RL
3. **Select Topology**: Different topologies affect synchronization difficulty
4. **Adjust Parameters**: Tune `dt`, `max_steps`, and reward parameters for your use case

## Examples

### Reinforcement Learning Training

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = KuramotoEnvironmentFactory.create_numpy_env(
    n_oscillators=6,
    coupling_mode="constant",  # Simpler action space
    reward_type="frequency_synchronization"
)

# Wrap for stable-baselines3
env = gym.wrappers.normalize.NormalizeObservation(env)

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Custom Reward Function

```python
# Create environment with custom reward
env = KuramotoEnvironmentFactory.create_numpy_env(
    n_oscillators=6,
    reward_type="combined",  # Uses both order parameter and phase coherence
    coupling_mode="dynamic"
)
```

### Multi-Agent Training

```python
# Create multi-agent environment
env = KuramotoEnvironmentFactory.create_pytorch_env(
    n_oscillators=6,
    n_agents=4,
    coupling_mode="constant",
    device="cuda"
)

# Train multiple agents in parallel
# (Implementation depends on your multi-agent RL framework)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or number of agents
2. **Poor Synchronization**: Check coupling strength range and topology
3. **Slow Training**: Use GPU and constant coupling mode for simpler problems
4. **Dimension Mismatch**: Ensure adjacency matrix size matches n_oscillators

### Performance Optimization

1. **Use PyTorch for Large Systems**: Better performance for many oscillators
2. **Choose Appropriate Topology**: Fully connected is easier to synchronize
3. **Tune Integration Parameters**: Smaller `dt` for accuracy, larger for speed
4. **Monitor Metrics**: Watch order parameter and reward curves

## Contributing

To extend the implementation:

1. **Add New Topologies**: Extend `TopologyGenerator` class
2. **Add New Coupling Matrices**: Extend `CouplingMatrixGenerator` class
3. **Add New Reward Functions**: Modify environment reward computation
4. **Add New Integration Methods**: Extend PyTorch environment integration

## References

- Original Kuramoto model: "Chemical Oscillations, Waves, and Turbulence"
- Reinforcement Learning: Various RL algorithms can be applied
- Network Science: Different topologies affect synchronization dynamics 