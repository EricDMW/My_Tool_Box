#!/usr/bin/env python3
"""
Comprehensive Implementation of Kuramoto Oscillator Environment

This implementation provides both NumPy and PyTorch versions of the Kuramoto oscillator
environment with support for dynamic and constant coupling modes.

Features:
- Dynamic coupling: Coupling strengths controlled by actions
- Constant coupling: Fixed coupling matrix provided at initialization
- Multiple reward types: order_parameter, phase_coherence, frequency_synchronization
- Custom network topologies
- GPU support for PyTorch version
- Multi-agent support for PyTorch version
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, Union
import gymnasium as gym
from gymnasium import spaces

import env_lib

# Import the environment classes
from env_lib import KuramotoOscillatorEnv
from env_lib import KuramotoOscillatorEnvTorch

class KuramotoEnvironmentFactory:
    """Factory class for creating Kuramoto oscillator environments."""
    
    @staticmethod
    def create_numpy_env(
        n_oscillators: int = 6,
        coupling_mode: str = "dynamic",
        constant_coupling_matrix: Optional[np.ndarray] = None,
        reward_type: str = "frequency_synchronization",
        target_frequency: float = 2.0,
        adj_matrix: Optional[np.ndarray] = None,
        **kwargs
    ) -> KuramotoOscillatorEnv:
        """
        Create a NumPy-based Kuramoto oscillator environment.
        
        Args:
            n_oscillators: Number of oscillators
            coupling_mode: "dynamic" or "constant"
            constant_coupling_matrix: Fixed coupling matrix for constant mode
            reward_type: Type of reward function
            target_frequency: Target frequency for frequency_synchronization reward
            adj_matrix: Custom adjacency matrix
            **kwargs: Additional environment parameters
        
        Returns:
            KuramotoOscillatorEnv instance
        """
        return KuramotoOscillatorEnv(
            n_oscillators=n_oscillators,
            coupling_mode=coupling_mode,
            constant_coupling_matrix=constant_coupling_matrix,
            reward_type=reward_type,
            target_frequency=target_frequency,
            adj_matrix=adj_matrix,
            **kwargs
        )
    
    @staticmethod
    def create_pytorch_env(
        n_oscillators: int = 6,
        n_agents: int = 1,
        coupling_mode: str = "dynamic",
        constant_coupling_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None,
        reward_type: str = "frequency_synchronization",
        target_frequency: float = 2.0,
        adj_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: str = "auto",
        **kwargs
    ) -> KuramotoOscillatorEnvTorch:
        """
        Create a PyTorch-based Kuramoto oscillator environment.
        
        Args:
            n_oscillators: Number of oscillators
            n_agents: Number of parallel agents
            coupling_mode: "dynamic" or "constant"
            constant_coupling_matrix: Fixed coupling matrix for constant mode
            reward_type: Type of reward function
            target_frequency: Target frequency for frequency_synchronization reward
            adj_matrix: Custom adjacency matrix
            device: "auto", "cpu", or "cuda"
            **kwargs: Additional environment parameters
        
        Returns:
            KuramotoOscillatorEnvTorch instance
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return KuramotoOscillatorEnvTorch(
            n_oscillators=n_oscillators,
            n_agents=n_agents,
            coupling_mode=coupling_mode,
            constant_coupling_matrix=constant_coupling_matrix,
            reward_type=reward_type,
            target_frequency=target_frequency,
            adj_matrix=adj_matrix,
            device=device,
            **kwargs
        )

class TopologyGenerator:
    """Utility class for generating different network topologies."""
    
    @staticmethod
    def fully_connected(n_oscillators: int) -> np.ndarray:
        """Create a fully connected topology."""
        adj_matrix = np.ones((n_oscillators, n_oscillators))
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix
    
    @staticmethod
    def ring(n_oscillators: int) -> np.ndarray:
        """Create a ring topology."""
        adj_matrix = np.zeros((n_oscillators, n_oscillators))
        for i in range(n_oscillators):
            adj_matrix[i, (i-1) % n_oscillators] = 1
            adj_matrix[i, (i+1) % n_oscillators] = 1
        return adj_matrix
    
    @staticmethod
    def star(n_oscillators: int) -> np.ndarray:
        """Create a star topology."""
        adj_matrix = np.zeros((n_oscillators, n_oscillators))
        adj_matrix[0, 1:] = 1  # Central oscillator
        adj_matrix[1:, 0] = 1  # Others connected to center
        return adj_matrix
    
    @staticmethod
    def two_clusters(n_oscillators: int) -> np.ndarray:
        """Create two disconnected clusters."""
        if n_oscillators % 2 != 0:
            raise ValueError("n_oscillators must be even for two clusters")
        
        half_n = n_oscillators // 2
        
        # Create a fully connected cluster
        cluster = np.ones((half_n, half_n)) - np.eye(half_n)
        
        # Create a block diagonal matrix with two such clusters
        adj_matrix = np.block([
            [cluster, np.zeros((half_n, half_n))],
            [np.zeros((half_n, half_n)), cluster]
        ])
        
        return adj_matrix
    
    @staticmethod
    def random(n_oscillators: int, connection_prob: float = 0.5) -> np.ndarray:
        """Create a random topology."""
        adj_matrix = np.random.random((n_oscillators, n_oscillators)) < connection_prob
        np.fill_diagonal(adj_matrix, 0)  # No self-connections
        # Ensure symmetry
        adj_matrix = (adj_matrix + adj_matrix.T) > 0
        return adj_matrix.astype(float)

class CouplingMatrixGenerator:
    """Utility class for generating different coupling matrices."""
    
    @staticmethod
    def uniform(n_oscillators: int, strength: float = 2.0) -> np.ndarray:
        """Create a uniform coupling matrix."""
        coupling_matrix = np.ones((n_oscillators, n_oscillators)) * strength
        np.fill_diagonal(coupling_matrix, 0)  # No self-coupling
        return coupling_matrix
    
    @staticmethod
    def distance_based(n_oscillators: int, base_strength: float = 2.0, decay: float = 0.5) -> np.ndarray:
        """Create a distance-based coupling matrix."""
        coupling_matrix = np.zeros((n_oscillators, n_oscillators))
        
        for i in range(n_oscillators):
            for j in range(n_oscillators):
                if i != j:
                    # Calculate distance (assuming oscillators are arranged in a circle)
                    distance = min(abs(i - j), n_oscillators - abs(i - j))
                    coupling_matrix[i, j] = base_strength * np.exp(-decay * distance)
        
        return coupling_matrix
    
    @staticmethod
    def hierarchical(n_oscillators: int, base_strength: float = 2.0) -> np.ndarray:
        """Create a hierarchical coupling matrix."""
        coupling_matrix = np.zeros((n_oscillators, n_oscillators))
        
        for i in range(n_oscillators):
            for j in range(n_oscillators):
                if i != j:
                    # Stronger coupling for oscillators closer to center
                    center_distance_i = abs(i - n_oscillators // 2)
                    center_distance_j = abs(j - n_oscillators // 2)
                    avg_center_distance = (center_distance_i + center_distance_j) / 2
                    coupling_matrix[i, j] = base_strength / (1 + avg_center_distance)
        
        return coupling_matrix

class EnvironmentTester:
    """Class for testing and evaluating environment performance."""
    
    def __init__(self, env, n_episodes: int = 10, max_steps: int = 200):
        self.env = env
        self.n_episodes = n_episodes
        self.max_steps = max_steps
    
    def run_random_policy(self) -> Dict[str, Any]:
        """Run random policy and collect statistics."""
        episode_rewards = []
        episode_order_params = []
        episode_lengths = []
        
        for episode in range(self.n_episodes):
            obs, info = self.env.reset(seed=episode)
            episode_reward = 0.0
            episode_order_param = 0.0
            step_count = 0
            
            for step in range(self.max_steps):
                action = self.env.action_space.sample()
                obs, reward, done, truncated, info = self.env.step(action)
                
                episode_reward += reward
                if 'order_parameter' in info:
                    if isinstance(info['order_parameter'], np.ndarray):
                        episode_order_param += info['order_parameter'][0]
                    else:
                        episode_order_param += info['order_parameter']
                
                step_count += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_order_params.append(episode_order_param / step_count)
            episode_lengths.append(step_count)
        
        return {
            'episode_rewards': episode_rewards,
            'episode_order_params': episode_order_params,
            'episode_lengths': episode_lengths,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_order_param': np.mean(episode_order_params),
            'std_order_param': np.std(episode_order_params),
            'avg_length': np.mean(episode_lengths)
        }
    
    def plot_results(self, results: Dict[str, Any], title: str = "Environment Performance"):
        """Plot the test results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot episode rewards
        ax1.plot(results['episode_rewards'], 'b-', alpha=0.7)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot episode order parameters
        ax2.plot(results['episode_order_params'], 'g-', alpha=0.7)
        ax2.set_title('Episode Order Parameters')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Order Parameter')
        ax2.grid(True, alpha=0.3)
        
        # Plot episode lengths
        ax3.plot(results['episode_lengths'], 'r-', alpha=0.7)
        ax3.set_title('Episode Lengths')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True, alpha=0.3)
        
        # Plot summary statistics
        ax4.text(0.1, 0.8, f"Avg Reward: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f"Avg Order Param: {results['avg_order_param']:.3f} ± {results['std_order_param']:.3f}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f"Avg Length: {results['avg_length']:.1f}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Summary Statistics')
        ax4.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig

def demo_dynamic_vs_constant():
    """Demonstrate the difference between dynamic and constant coupling modes."""
    print("=== Dynamic vs Constant Coupling Comparison ===")
    
    # Create topologies and coupling matrices
    n_oscillators = 6
    adj_matrix = TopologyGenerator.two_clusters(n_oscillators)
    constant_coupling_matrix = CouplingMatrixGenerator.uniform(n_oscillators, strength=2.0)
    
    # Test NumPy environments
    print("\nTesting NumPy environments...")
    
    # Dynamic coupling
    env_numpy_dynamic = KuramotoEnvironmentFactory.create_numpy_env(
        n_oscillators=n_oscillators,
        coupling_mode="dynamic",
        adj_matrix=adj_matrix,
        max_steps=200
    )
    
    tester_dynamic = EnvironmentTester(env_numpy_dynamic, n_episodes=5)
    results_dynamic = tester_dynamic.run_random_policy()
    
    # Constant coupling
    env_numpy_constant = KuramotoEnvironmentFactory.create_numpy_env(
        n_oscillators=n_oscillators,
        coupling_mode="constant",
        constant_coupling_matrix=constant_coupling_matrix,
        adj_matrix=adj_matrix,
        max_steps=200
    )
    
    tester_constant = EnvironmentTester(env_numpy_constant, n_episodes=5)
    results_constant = tester_constant.run_random_policy()
    
    # Print comparison
    print(f"\nNumPy Dynamic Coupling:")
    print(f"  Avg Reward: {results_dynamic['avg_reward']:.3f} ± {results_dynamic['std_reward']:.3f}")
    print(f"  Avg Order Param: {results_dynamic['avg_order_param']:.3f} ± {results_dynamic['std_order_param']:.3f}")
    print(f"  Action Space: {env_numpy_dynamic.action_space.shape}")
    
    print(f"\nNumPy Constant Coupling:")
    print(f"  Avg Reward: {results_constant['avg_reward']:.3f} ± {results_constant['std_reward']:.3f}")
    print(f"  Avg Order Param: {results_constant['avg_order_param']:.3f} ± {results_constant['std_order_param']:.3f}")
    print(f"  Action Space: {env_numpy_constant.action_space.shape}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot rewards comparison
    ax1.plot(results_dynamic['episode_rewards'], 'b-', label='Dynamic', alpha=0.7)
    ax1.plot(results_constant['episode_rewards'], 'r-', label='Constant', alpha=0.7)
    ax1.set_title('Episode Rewards Comparison')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot order parameters comparison
    ax2.plot(results_dynamic['episode_order_params'], 'b-', label='Dynamic', alpha=0.7)
    ax2.plot(results_constant['episode_order_params'], 'r-', label='Constant', alpha=0.7)
    ax2.set_title('Order Parameters Comparison')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Order Parameter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dynamic_vs_constant_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Clean up
    env_numpy_dynamic.close()
    env_numpy_constant.close()

def demo_pytorch_multi_agent():
    """Demonstrate PyTorch multi-agent environment."""
    print("=== PyTorch Multi-Agent Demo ===")
    
    n_oscillators = 6
    n_agents = 4
    adj_matrix = TopologyGenerator.ring(n_oscillators)
    constant_coupling_matrix = CouplingMatrixGenerator.distance_based(n_oscillators)
    
    # Create multi-agent environment
    env = KuramotoEnvironmentFactory.create_pytorch_env(
        n_oscillators=n_oscillators,
        n_agents=n_agents,
        coupling_mode="constant",
        constant_coupling_matrix=constant_coupling_matrix,
        adj_matrix=adj_matrix,
        max_steps=100,
        render_mode="human"
    )
    
    print(f"Environment created with {n_agents} agents")
    print(f"Action space: {env.action_space.shape}")
    print(f"Observation space: {env.observation_space.shape}")
    
    # Run a few steps with visualization
    obs, info = env.reset(seed=123)
    
    for step in range(50):
        # Random actions for all agents
        actions = [env.action_space.sample() for _ in range(n_agents)]
        
        # Take step (PyTorch env handles multi-agent internally)
        obs, reward, done, truncated, info = env.step(actions[0])  # Use first agent's action
        
        if step % 10 == 0:
            env.render()
            plt.pause(0.01)
            print(f"Step {step}: Reward = {reward:.4f}, Order Param = {info['order_parameter'][0]:.3f}")
        
        if done:
            break
    
    env.close()
    print("Multi-agent demo finished.")

def demo_different_topologies():
    """Demonstrate different network topologies."""
    print("=== Different Topologies Demo ===")
    
    n_oscillators = 8
    topologies = {
        'Fully Connected': TopologyGenerator.fully_connected(n_oscillators),
        'Ring': TopologyGenerator.ring(n_oscillators),
        'Star': TopologyGenerator.star(n_oscillators),
        'Two Clusters': TopologyGenerator.two_clusters(n_oscillators),
        'Random': TopologyGenerator.random(n_oscillators, connection_prob=0.3)
    }
    
    results = {}
    
    for name, adj_matrix in topologies.items():
        print(f"\nTesting {name} topology...")
        
        env = KuramotoEnvironmentFactory.create_numpy_env(
            n_oscillators=n_oscillators,
            coupling_mode="dynamic",
            adj_matrix=adj_matrix,
            max_steps=150
        )
        
        tester = EnvironmentTester(env, n_episodes=3)
        results[name] = tester.run_random_policy()
        
        env.close()
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot average rewards
    names = list(results.keys())
    avg_rewards = [results[name]['avg_reward'] for name in names]
    std_rewards = [results[name]['std_reward'] for name in names]
    
    ax1.bar(names, avg_rewards, yerr=std_rewards, alpha=0.7)
    ax1.set_title('Average Rewards by Topology')
    ax1.set_ylabel('Average Reward')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot average order parameters
    avg_order_params = [results[name]['avg_order_param'] for name in names]
    std_order_params = [results[name]['std_order_param'] for name in names]
    
    ax2.bar(names, avg_order_params, yerr=std_order_params, alpha=0.7)
    ax2.set_title('Average Order Parameters by Topology')
    ax2.set_ylabel('Average Order Parameter')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('topology_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    try:
        print("Kuramoto Oscillator Environment Implementation")
        print("=" * 50)
        
        # Run demos
        demo_dynamic_vs_constant()
        demo_pytorch_multi_agent()
        demo_different_topologies()
        
        print("\nAll demos completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc() 