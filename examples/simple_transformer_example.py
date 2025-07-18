"""
Simple Transformer Example for Reinforcement Learning

This example shows how to use transformer networks from the neural_toolkit
for a simple reinforcement learning task.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import from our toolkit
from toolkit.neural_toolkit import TransformerPolicyNetwork, NetworkUtils


def create_simple_environment():
    """Create a simple environment that provides sequential observations"""
    
    class SimpleSeqEnv:
        def __init__(self, seq_length=5, obs_dim=4, action_dim=3):
            self.seq_length = seq_length
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.reset()
        
        def reset(self):
            """Reset environment"""
            self.step_count = 0
            self.max_steps = 20
            # Create a sequence of observations
            self.obs_sequence = []
            for _ in range(self.seq_length):
                obs = np.random.randn(self.obs_dim)
                self.obs_sequence.append(obs)
            return self.get_obs()
        
        def step(self, action):
            """Take action and return (obs, reward, done, info)"""
            self.step_count += 1
            
            # Simple reward: positive reward for action 0, negative for others
            reward = 1.0 if action == 0 else -0.1
            
            # Update observation sequence
            new_obs = np.random.randn(self.obs_dim)
            self.obs_sequence.append(new_obs)
            self.obs_sequence.pop(0)  # Remove oldest observation
            
            done = self.step_count >= self.max_steps
            
            return self.get_obs(), reward, done, {}
        
        def get_obs(self):
            """Get current observation sequence"""
            return np.array(self.obs_sequence)
    
    return SimpleSeqEnv()


def train_transformer_policy():
    """Train a transformer policy on the simple environment"""
    
    print("🚀 Simple Transformer RL Example")
    print("=" * 40)
    
    # Set random seed for reproducibility
    torch.manual_seed(1997)
    np.random.seed(1997)
    
    # Configuration
    obs_dim = 4
    action_dim = 3
    seq_length = 5
    d_model = 64
    nhead = 4
    num_layers = 2
    
    print(f"📋 Configuration:")
    print(f"   Observation dimension: {obs_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Model dimension: {d_model}")
    print(f"   Number of heads: {nhead}")
    print(f"   Number of layers: {num_layers}")
    print()
    
    # Create environment
    env = create_simple_environment()
    
    # Create transformer policy network
    policy = TransformerPolicyNetwork(
        input_dim=obs_dim,
        output_dim=action_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 2,
        dropout=0.1,
        fc_dims=[d_model // 2],
        activation='relu',
        device='cpu'  # Use CPU for simplicity
    )
    
    # Initialize weights
    NetworkUtils.initialize_weights(policy, method='xavier_uniform')
    
    print(f"🧠 Transformer Policy Network:")
    print(f"   Parameters: {NetworkUtils.count_parameters(policy):,}")
    print()
    
    # Setup optimizer
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    
    # Training loop
    num_episodes = 10000
    episode_rewards = []
    
    print("🎯 Starting training...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        
        while True:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # (1, seq_len, obs_dim)
            
            # Get action probabilities
            action_logits = policy(obs_tensor)  # (1, action_dim)
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Sample action
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            # Take action in environment
            obs, reward, done, _ = env.step(action.item())
            total_reward += reward
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        
        # Simple policy gradient update
        if episode % 10 == 0:
            # Compute loss (simplified policy gradient)
            loss = -torch.log(action_probs[0, action.item()]) * total_reward
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode:3d}: Reward = {total_reward:6.2f}, Avg (last 10) = {avg_reward:6.2f}")
    
    print(f"\n✅ Training completed!")
    
    # Final evaluation
    final_rewards = []
    for _ in range(10):
        obs = env.reset()
        total_reward = 0
        
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_logits = policy(obs_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.argmax(action_probs, dim=-1).item()
            
            obs, reward, done, _ = env.step(int(action))
            total_reward += reward
            
            if done:
                break
        
        final_rewards.append(total_reward)
    
    print(f"\n🏆 Final Performance:")
    print(f"   Mean Reward: {np.mean(final_rewards):.2f}")
    print(f"   Std Reward: {np.std(final_rewards):.2f}")
    print(f"   Min Reward: {np.min(final_rewards):.2f}")
    print(f"   Max Reward: {np.max(final_rewards):.2f}")
    
    # Plot training progress
    plot_training_progress(episode_rewards)
    
    return policy, env


def plot_training_progress(rewards):
    """Plot training progress"""
    plt.figure(figsize=(10, 6))
    
    # Plot individual episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, color='blue', label='Episode Reward')
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Episodic Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    window_size = 10
    moving_avg = []
    for i in range(len(rewards)):
        if i < window_size - 1:
            moving_avg.append(np.mean(rewards[:i+1]))
        else:
            moving_avg.append(np.mean(rewards[i-window_size+1:i+1]))
    
    plt.plot(moving_avg, color='red', linewidth=2, label=f'{window_size}-Episode Moving Average')
    plt.title('Constraint Violation (times per episode)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_transformer_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Training plot saved as 'simple_transformer_training.png'")


def demonstrate_attention_weights(policy, env):
    """Demonstrate how to extract attention weights from the transformer"""
    print(f"\n🔍 Demonstrating Attention Weights:")
    print("-" * 40)
    
    # Get a sample observation
    obs = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    
    # Forward pass to get action
    action_logits = policy(obs_tensor)
    action_probs = F.softmax(action_logits, dim=-1)
    action = torch.argmax(action_probs, dim=-1).item()
    
    print(f"Sample observation sequence shape: {obs.shape}")
    print(f"Selected action: {action}")
    print(f"Action probabilities: {action_probs[0].detach().numpy()}")
    
    # Note: To extract actual attention weights, you would need to modify
    # the transformer implementation to return attention weights
    print("💡 Note: To extract attention weights, modify the transformer")
    print("   to return attention weights during forward pass")


def compare_with_mlp():
    """Compare transformer with MLP on the same task"""
    print(f"\n⚖️  Comparing Transformer vs MLP:")
    print("-" * 40)
    
    from toolkit.neural_toolkit import MLPPolicyNetwork
    
    # Create both networks
    obs_dim = 4
    action_dim = 3
    seq_length = 5
    
    # Transformer policy
    transformer_policy = TransformerPolicyNetwork(
        input_dim=obs_dim,
        output_dim=action_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        device='cpu'
    )
    
    # MLP policy (flattened sequence)
    mlp_policy = MLPPolicyNetwork(
        input_dim=obs_dim * seq_length,  # Flattened sequence
        output_dim=action_dim,
        hidden_dims=[128, 64],
        device='cpu'
    )
    
    print(f"Transformer parameters: {NetworkUtils.count_parameters(transformer_policy):,}")
    print(f"MLP parameters: {NetworkUtils.count_parameters(mlp_policy):,}")
    
    # Test forward pass
    env = create_simple_environment()
    obs = env.reset()
    
    # Transformer forward pass
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # (1, seq_len, obs_dim)
    transformer_output = transformer_policy(obs_tensor)
    
    # MLP forward pass (flattened)
    obs_flat = obs.flatten()  # (seq_len * obs_dim,)
    obs_flat_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)  # (1, seq_len * obs_dim)
    mlp_output = mlp_policy(obs_flat_tensor)
    
    print(f"Transformer output shape: {transformer_output.shape}")
    print(f"MLP output shape: {mlp_output.shape}")
    print(f"Transformer action: {torch.argmax(transformer_output, dim=-1).item()}")
    print(f"MLP action: {torch.argmax(mlp_output, dim=-1).item()}")


if __name__ == "__main__":
    # Run the simple transformer example
    policy, env = train_transformer_policy()
    
    # Demonstrate attention weights
    demonstrate_attention_weights(policy, env)
    
    # Compare with MLP
    compare_with_mlp()
    
    print(f"\n🎉 Simple Transformer Example Completed!")
    print(f"\n💡 Key Points:")
    print(f"   • Transformer networks can handle sequential observations")
    print(f"   • Attention mechanisms help capture temporal dependencies")
    print(f"   • The toolkit provides easy-to-use transformer implementations")
    print(f"   • Transformers are more parameter-efficient than MLPs for sequences") 