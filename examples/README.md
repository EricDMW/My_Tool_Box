# Neural Toolkit Examples

This directory contains comprehensive examples demonstrating how to use the neural_toolkit for various reinforcement learning tasks.

## 🚀 Quick Start

### Prerequisites
```bash
# Install required dependencies
pip install torch numpy matplotlib
```

### Running Examples
```bash
# Run the simple transformer example
python simple_transformer_example.py

# Run the comprehensive transformer example
python transformer_rl_example.py
```

## 📚 Available Examples

### 1. Simple Transformer Example (`simple_transformer_example.py`)

**Perfect for beginners!** This example demonstrates the core concepts of using transformer networks for reinforcement learning.

**Features:**
- ✅ Simple sequential environment
- ✅ Transformer policy network setup
- ✅ Basic training loop with policy gradients
- ✅ Performance visualization
- ✅ Comparison with MLP networks
- ✅ Easy to understand and modify

**What you'll learn:**
- How to create transformer networks using the toolkit
- How to handle sequential observations
- Basic reinforcement learning training
- Performance comparison between architectures

**Expected Output:**
```
🚀 Simple Transformer RL Example
========================================
📋 Configuration:
   Observation dimension: 4
   Action dimension: 3
   Sequence length: 5
   Model dimension: 64
   Number of heads: 4
   Number of layers: 2

🧠 Transformer Policy Network:
   Parameters: 45,123

🎯 Starting training...
Episode   0: Reward =  15.20, Avg (last 10) =  15.20
Episode  10: Reward =  16.80, Avg (last 10) =  16.45
...
```

### 2. Comprehensive Transformer Example (`transformer_rl_example.py`)

**Advanced example** with full PPO-style training and sophisticated environment.

**Features:**
- ✅ Complex sequential environment with temporal dependencies
- ✅ Complete PPO-style training implementation
- ✅ GAE (Generalized Advantage Estimation)
- ✅ Experience collection and replay
- ✅ Comprehensive evaluation and monitoring
- ✅ Advanced visualization and analysis
- ✅ Production-ready training loop

**What you'll learn:**
- Advanced reinforcement learning techniques
- PPO algorithm implementation
- Experience buffer management
- Performance monitoring and debugging
- Real-world RL training practices

**Expected Output:**
```
🚀 Starting Transformer RL Training Example
============================================================
📋 Configuration:
   obs_dim: 8
   action_dim: 4
   seq_length: 10
   d_model: 128
   nhead: 8
   num_layers: 4
   dim_feedforward: 512
   dropout: 0.1
   lr: 0.0003
   device: cuda

🧠 Agent created with:
   Policy parameters: 1,234,567
   Value parameters: 987,654
   Total parameters: 2,222,221

🎯 Starting training...
Iteration   0/50: Reward =  12.45 ±  3.21, Policy Loss =  0.234, Value Loss =  0.156
Iteration   5/50: Reward =  18.67 ±  2.89, Policy Loss =  0.189, Value Loss =  0.123
...
```

## 🎯 Example Use Cases

### Sequential Decision Making
Transformers excel at tasks where the agent needs to consider the history of observations:

```python
# Example: Robot navigation with memory
obs_sequence = [
    [sensor_reading_1, position_1, velocity_1],
    [sensor_reading_2, position_2, velocity_2],
    [sensor_reading_3, position_3, velocity_3],
    # ... more timesteps
]
action = transformer_policy(obs_sequence)
```

### Multi-Agent Coordination
Transformers can handle complex multi-agent scenarios:

```python
# Example: Multi-agent communication
agent_observations = [
    [agent1_obs, agent2_obs, agent3_obs],  # timestep 1
    [agent1_obs, agent2_obs, agent3_obs],  # timestep 2
    # ... more timesteps
]
coordinated_action = transformer_policy(agent_observations)
```

### Natural Language RL
Transformers are perfect for RL tasks involving text:

```python
# Example: Text-based game playing
text_sequence = [
    "You are in a dark room.",
    "There is a door to the north.",
    "You hear footsteps approaching.",
    # ... more text observations
]
action = transformer_policy(encode_text(text_sequence))
```

## 🔧 Customization Guide

### Modifying Network Architecture

```python
# Change transformer parameters
policy = TransformerPolicyNetwork(
    input_dim=10,           # Your observation dimension
    output_dim=5,           # Your action dimension
    d_model=256,            # Larger model for complex tasks
    nhead=16,               # More attention heads
    num_layers=8,           # Deeper network
    dim_feedforward=1024,   # Larger feedforward layers
    dropout=0.2,            # More dropout for regularization
    device='cuda'           # Use GPU for faster training
)
```

### Creating Custom Environments

```python
class MyCustomEnvironment:
    def __init__(self, seq_length=10):
        self.seq_length = seq_length
        self.reset()
    
    def reset(self):
        # Initialize your environment
        self.obs_history = []
        return self.get_observation()
    
    def step(self, action):
        # Implement your environment dynamics
        reward = self.compute_reward(action)
        done = self.is_episode_done()
        return self.get_observation(), reward, done, {}
    
    def get_observation(self):
        # Return sequence of observations
        return np.array(self.obs_history[-self.seq_length:])
```

### Training Configuration

```python
# Adjust training parameters
config = {
    'lr': 1e-4,              # Learning rate
    'gamma': 0.99,           # Discount factor
    'gae_lambda': 0.95,      # GAE parameter
    'clip_epsilon': 0.2,     # PPO clip parameter
    'value_loss_coef': 0.5,  # Value loss coefficient
    'entropy_coef': 0.01,    # Entropy coefficient for exploration
}
```

## 📊 Performance Tips

### 1. Sequence Length Optimization
```python
# For short-term dependencies
seq_length = 5-10

# For long-term dependencies
seq_length = 20-50

# For very long sequences, consider hierarchical approaches
seq_length = 100+  # Use attention over chunks
```

### 2. Model Size Guidelines
```python
# Small tasks (simple environments)
d_model = 64-128
nhead = 4-8
num_layers = 2-4

# Medium tasks (moderate complexity)
d_model = 128-256
nhead = 8-16
num_layers = 4-8

# Large tasks (complex environments)
d_model = 256-512
nhead = 16-32
num_layers = 8-12
```

### 3. Training Stability
```python
# Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# Normalize advantages
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Use learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory
```python
# Reduce batch size or sequence length
config['seq_length'] = 5  # Instead of 20
config['d_model'] = 64    # Instead of 256

# Use gradient accumulation
accumulation_steps = 4
```

#### 2. Slow Training
```python
# Use GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reduce model complexity
config['num_layers'] = 2  # Instead of 8
config['nhead'] = 4       # Instead of 16
```

#### 3. Poor Performance
```python
# Increase exploration
config['entropy_coef'] = 0.05  # Instead of 0.01

# Adjust learning rate
config['lr'] = 1e-4  # Try different values

# Increase training time
num_iterations = 200  # Instead of 50
```

## 📈 Expected Results

### Simple Example
- **Training Time**: 1-5 minutes on CPU
- **Final Reward**: 15-20 (depending on environment)
- **Convergence**: 50-100 episodes

### Comprehensive Example
- **Training Time**: 10-30 minutes on GPU
- **Final Reward**: 20-30 (depending on environment)
- **Convergence**: 100-200 iterations

## 🤝 Contributing Examples

We welcome new examples! When contributing:

1. **Keep it simple**: Start with basic concepts
2. **Add documentation**: Explain what the example demonstrates
3. **Include visualization**: Show training progress and results
4. **Test thoroughly**: Ensure the example runs without errors
5. **Follow the style**: Use consistent formatting and structure

## 📞 Support

If you encounter issues with the examples:

1. Check the troubleshooting section above
2. Verify your dependencies are installed correctly
3. Try the simple example first
4. Open an issue on GitHub with:
   - Error message
   - Your configuration
   - Expected vs actual behavior

---

**Happy Learning! 🚀** 