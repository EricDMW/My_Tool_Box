"""
Transformer Reinforcement Learning Example

This example demonstrates how to use transformer networks for reinforcement learning
with sequential data. The example includes:
- Transformer policy network setup
- Custom environment with sequential observations
- Complete training loop with PPO-style updates
- Performance monitoring and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import random
from collections import deque
import time

# Import from our toolkit
from toolkit.neural_toolkit import TransformerPolicyNetwork, TransformerValueNetwork
from toolkit.neural_toolkit import NetworkUtils


class SequentialEnvironment:
    """
    A simple sequential environment that provides sequential observations
    to test transformer-based RL agents.
    """
    
    def __init__(self, seq_length: int = 10, obs_dim: int = 8, action_dim: int = 4):
        self.seq_length = seq_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation sequence"""
        self.step_count = 0
        self.max_steps = 100
        self.observation_history = deque(maxlen=self.seq_length)
        
        # Initialize with random observations
        for _ in range(self.seq_length):
            obs = np.random.randn(self.obs_dim)
            self.observation_history.append(obs)
            
        return self.get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return (observation, reward, done, info)"""
        self.step_count += 1
        
        # Simulate environment dynamics
        # Reward depends on action consistency and observation patterns
        current_obs = np.array(self.observation_history[-1])
        
        # Simple reward function: reward for taking action 0 when obs[0] > 0
        if action == 0 and current_obs[0] > 0:
            reward = 1.0
        elif action == 1 and current_obs[1] > 0:
            reward = 0.5
        elif action == 2 and current_obs[2] > 0:
            reward = 0.3
        else:
            reward = -0.1
            
        # Add some randomness to make it more interesting
        reward += np.random.normal(0, 0.1)
        
        # Generate new observation based on action
        new_obs = current_obs.copy()
        new_obs[action] += np.random.normal(0, 0.5)
        new_obs = np.clip(new_obs, -2, 2)  # Clip to reasonable range
        
        # Add to history
        self.observation_history.append(new_obs)
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        info = {
            'step': self.step_count,
            'action': action,
            'reward': reward
        }
        
        return self.get_observation(), reward, done, info
    
    def get_observation(self) -> np.ndarray:
        """Get current observation sequence"""
        return np.array(list(self.observation_history))


class TransformerRLAgent:
    """
    Transformer-based reinforcement learning agent using our toolkit.
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 seq_length: int = 10,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 lr: float = 3e-4,
                 device: str = 'cuda'):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.device = device
        
        # Create transformer policy network
        self.policy = TransformerPolicyNetwork(
            input_dim=obs_dim,
            output_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            fc_dims=[d_model // 2],
            activation='relu',
            device=device
        )
        
        # Create transformer value network
        self.value = TransformerValueNetwork(
            input_dim=obs_dim,
            output_dim=1,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            fc_dims=[d_model // 2],
            activation='relu',
            device=device
        )
        
        # Move networks to device
        self.policy = self.policy.to(device)
        self.value = self.value.to(device)
        
        # Verify networks are on correct device
        print(f"🔧 Networks moved to device: {device}")
        print(f"   Policy network device: {next(self.policy.parameters()).device}")
        print(f"   Value network device: {next(self.value.parameters()).device}")
        print()
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        # Initialize weights
        NetworkUtils.initialize_weights(self.policy, method='xavier_uniform')
        NetworkUtils.initialize_weights(self.value, method='xavier_uniform')
        
        # Training parameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        # Experience buffer
        self.experience_buffer = []
        
    def get_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Get action from policy network.
        Returns: (action, log_prob, value)
        """
        with torch.no_grad():
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, seq_len, obs_dim)
            
            # Verify tensor is on correct device
            if obs_tensor.device != next(self.policy.parameters()).device:
                print(f"⚠️  Device mismatch: tensor on {obs_tensor.device}, policy on {next(self.policy.parameters()).device}")
                obs_tensor = obs_tensor.to(next(self.policy.parameters()).device)
            
            # Get action probabilities
            action_probs = self.policy(obs_tensor)  # (1, action_dim)
            action_probs = F.softmax(action_probs, dim=-1)
            
            # Sample action
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Get value estimate
            value = self.value(obs_tensor)
            
            return int(action.item()), log_prob.item(), float(value.item())
    
    def collect_experience(self, env: SequentialEnvironment, num_episodes: int = 10) -> List[Dict]:
        """Collect experience from environment"""
        all_experiences = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_experiences = []
            episode_rewards = []
            
            while True:
                action, log_prob, value = self.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                
                experience = {
                    'obs': obs.copy(),
                    'action': action,
                    'reward': reward,
                    'log_prob': log_prob,
                    'value': value,
                    'done': done
                }
                episode_experiences.append(experience)
                episode_rewards.append(reward)
                
                if done:
                    break
                    
                obs = next_obs
            
            # Calculate returns and advantages
            returns = self._compute_returns(episode_rewards)
            advantages = self._compute_advantages(episode_experiences, returns)
            
            # Add returns and advantages to experiences
            for i, exp in enumerate(episode_experiences):
                exp['return'] = returns[i]
                exp['advantage'] = advantages[i]
                all_experiences.append(exp)
        
        return all_experiences
    
    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns"""
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns
    
    def _compute_advantages(self, experiences: List[Dict], returns: List[float]) -> List[float]:
        """Compute GAE advantages"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(experiences))):
            if i == len(experiences) - 1:
                next_value = 0
            else:
                next_value = experiences[i + 1]['value']
            
            delta = experiences[i]['reward'] + self.gamma * next_value - experiences[i]['value']
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update_policy(self, experiences: List[Dict], num_epochs: int = 4) -> Dict[str, float]:
        """Update policy and value networks using PPO-style updates"""
        if not experiences:
            return {}
        
        # Convert experiences to tensors
        obs_batch = torch.FloatTensor([exp['obs'] for exp in experiences]).to(self.device)
        action_batch = torch.LongTensor([exp['action'] for exp in experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in experiences]).to(self.device)
        returns_batch = torch.FloatTensor([exp['return'] for exp in experiences]).to(self.device)
        advantages_batch = torch.FloatTensor([exp['advantage'] for exp in experiences]).to(self.device)
        
        # Verify all tensors are on correct device
        policy_device = next(self.policy.parameters()).device
        if obs_batch.device != policy_device:
            print(f"⚠️  Device mismatch in update_policy: tensors on {obs_batch.device}, policy on {policy_device}")
            obs_batch = obs_batch.to(policy_device)
            action_batch = action_batch.to(policy_device)
            old_log_probs = old_log_probs.to(policy_device)
            returns_batch = returns_batch.to(policy_device)
            advantages_batch = advantages_batch.to(policy_device)
        
        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(num_epochs):
            # Get current policy probabilities
            action_probs = self.policy(obs_batch)
            action_probs = F.softmax(action_probs, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            
            # Compute new log probabilities
            new_log_probs = action_dist.log_prob(action_batch)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.value(obs_batch).squeeze()
            value_loss = F.mse_loss(values, returns_batch)
            
            # Entropy loss for exploration
            entropy = action_dist.entropy().mean()
            entropy_loss = -entropy
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())
        
        return {
            'policy_loss': float(np.mean(policy_losses)),
            'value_loss': float(np.mean(value_losses)),
            'entropy': float(np.mean(entropy_losses))
        }
    
    def evaluate(self, env: SequentialEnvironment, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate agent performance"""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action, _, _ = self.get_action(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards))
        }


def train_transformer_agent():
    """Main training function"""
    print("🚀 Starting Transformer RL Training Example")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Configuration
    config = {
        'obs_dim': 8,
        'action_dim': 4,
        'seq_length': 10,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'lr': 3e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    if config['device'] == 'cuda':
        print("🚀 Using CUDA for training")
    else:
        print("💻 Using CPU for training (CUDA not available)")
    print()
    
    # Create environment and agent
    env = SequentialEnvironment(
        seq_length=config['seq_length'],
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim']
    )
    
    agent = TransformerRLAgent(**config)
    
    print(f"🧠 Agent created with:")
    print(f"   Policy parameters: {NetworkUtils.count_parameters(agent.policy):,}")
    print(f"   Value parameters: {NetworkUtils.count_parameters(agent.value):,}")
    print(f"   Total parameters: {NetworkUtils.count_parameters(agent.policy) + NetworkUtils.count_parameters(agent.value):,}")
    print()
    
    # Training loop
    num_iterations = 50
    episodes_per_iteration = 5
    evaluation_interval = 5
    
    training_rewards = []
    evaluation_rewards = []
    policy_losses = []
    value_losses = []
    
    print("🎯 Starting training...")
    start_time = time.time()
    
    for iteration in range(num_iterations):
        # Collect experience
        experiences = agent.collect_experience(env, episodes_per_iteration)
        
        # Update policy
        update_info = agent.update_policy(experiences)
        
        # Record metrics
        if update_info:
            policy_losses.append(update_info['policy_loss'])
            value_losses.append(update_info['value_loss'])
        
        # Evaluate periodically
        if iteration % evaluation_interval == 0:
            eval_info = agent.evaluate(env, num_episodes=3)
            evaluation_rewards.append(eval_info['mean_reward'])
            
            print(f"Iteration {iteration:3d}/{num_iterations}: "
                  f"Reward = {eval_info['mean_reward']:6.2f} ± {eval_info['std_reward']:5.2f}, "
                  f"Policy Loss = {update_info.get('policy_loss', 0):6.3f}, "
                  f"Value Loss = {update_info.get('value_loss', 0):6.3f}")
        
        # Record training reward (average of collected episodes)
        if experiences:
            avg_reward = np.mean([exp['reward'] for exp in experiences])
            training_rewards.append(avg_reward)
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.1f} seconds")
    
    # Final evaluation
    final_eval = agent.evaluate(env, num_episodes=10)
    print(f"\n🏆 Final Performance:")
    print(f"   Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"   Min Reward: {final_eval['min_reward']:.2f}")
    print(f"   Max Reward: {final_eval['max_reward']:.2f}")
    print(f"   Mean Episode Length: {final_eval['mean_length']:.1f}")
    
    # Plot results
    plot_training_results(training_rewards, evaluation_rewards, policy_losses, value_losses)
    
    return agent, env


def plot_training_results(training_rewards, evaluation_rewards, policy_losses, value_losses):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Transformer RL Training Results', fontsize=16)
    
    # Training rewards
    axes[0, 0].plot(training_rewards, alpha=0.7, color='blue')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Evaluation rewards
    if evaluation_rewards:
        eval_iterations = list(range(0, len(training_rewards), 5))
        axes[0, 1].plot(eval_iterations, evaluation_rewards, 'o-', color='red')
        axes[0, 1].set_title('Evaluation Rewards')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Mean Reward')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Policy loss
    if policy_losses:
        axes[1, 0].plot(policy_losses, color='green')
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Value loss
    if value_losses:
        axes[1, 1].plot(value_losses, color='orange')
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformer_rl_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Training plots saved as 'transformer_rl_training_results.png'")


def demonstrate_agent_behavior(agent, env, num_episodes=3):
    """Demonstrate trained agent behavior"""
    print(f"\n🎭 Demonstrating Agent Behavior ({num_episodes} episodes):")
    print("-" * 50)
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        actions_taken = []
        
        print(f"Episode {episode + 1}:")
        
        while True:
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            actions_taken.append(action)
            
            if done:
                break
            
            obs = next_obs
        
        print(f"   Steps: {steps:3d}, Total Reward: {total_reward:6.2f}")
        print(f"   Actions: {actions_taken[:10]}{'...' if len(actions_taken) > 10 else ''}")
        print()


if __name__ == "__main__":
    # Run the training example
    agent, env = train_transformer_agent()
    
    # Demonstrate the trained agent
    demonstrate_agent_behavior(agent, env)
    
    print("🎉 Transformer RL Example Completed!")
    print("\n💡 Key Takeaways:")
    print("   • Transformer networks can effectively handle sequential RL problems")
    print("   • Attention mechanisms help capture temporal dependencies")
    print("   • PPO-style updates work well with transformer policies")
    print("   • The toolkit provides easy-to-use transformer implementations") 