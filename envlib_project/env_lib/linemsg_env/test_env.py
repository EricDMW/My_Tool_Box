"""
Test script for the new Line Message environment.

This script demonstrates how to use the new line message environment with
different configurations and parameters.
"""

import numpy as np
import gymnasium as gym
from linemsg_env import LineMsgEnv


def test_basic_usage():
    """Test basic environment usage."""
    print("=== Testing Basic Environment Usage ===")
    
    # Create environment with default parameters
    env = LineMsgEnv(num_agents=5, render_mode="human")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Number of agents: {env.num_agents}")
    
    # Take a few random steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Reward = {reward:.3f}, Terminated = {terminated}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Basic test completed!\n")


def test_different_configurations():
    """Test environment with different configurations."""
    print("=== Testing Different Configurations ===")
    
    configs = [
        {"num_agents": 3, "name": "3 agents"},
        {"num_agents": 8, "name": "8 agents"},
        {"num_agents": 5, "n_obs_neighbors": 2, "name": "5 agents, 2-hop neighbors"},
        {"num_agents": 6, "max_iter": 20, "name": "6 agents, short episode"},
    ]
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        # Remove 'name' from config for environment creation
        env_config = {k: v for k, v in config.items() if k != 'name'}
        env = LineMsgEnv(**env_config, render_mode="human")
        
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        print(f"  Number of agents: {env.num_agents}")
        
        # Take a few steps
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"  Total reward over 5 steps: {total_reward:.3f}")
        env.close()
    
    print("Configuration tests completed!\n")


def test_message_passing():
    """Test message passing dynamics."""
    print("=== Testing Message Passing Dynamics ===")
    
    # Create environment
    env = LineMsgEnv(num_agents=6, render_mode="human")
    
    obs, info = env.reset()
    if env.state is not None:
        print(f"Initial state: {env.state[env.n_obs_nghbr:-env.n_obs_nghbr]}")
    else:
        print("Initial state is not initialized.")
    
    # Track message propagation
    message_counts = []
    
    for step in range(15):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Count messages in the system
        if env.state is not None:
            messages = np.sum(env.state[env.n_obs_nghbr:-env.n_obs_nghbr] == 1)
            message_counts.append(messages)
        else:
            messages = 0
            message_counts.append(messages)
        
        if step % 5 == 0:
            print(f"Step {step}: Messages = {messages}, Reward = {reward:.3f}")
            if env.state is not None:
                print(f"  State: {env.state[env.n_obs_nghbr:-env.n_obs_nghbr]}")
            else:
                print("  State is not initialized.")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print(f"Message count history: {message_counts}")
    env.close()
    print("Message passing test completed!\n")


def test_neighborhood_observation():
    """Test different observation neighborhood sizes."""
    print("=== Testing Observation Neighborhood ===")
    
    neighborhood_configs = [
        {"n_obs_neighbors": 1, "name": "1-hop neighbors (minimum)"},
        {"n_obs_neighbors": 2, "name": "2-hop neighbors"},
        {"n_obs_neighbors": 3, "name": "3-hop neighbors"},
    ]
    
    for config in neighborhood_configs:
        print(f"\nTesting: {config['name']}")
        
        # Remove 'name' from config for environment creation
        env_config = {k: v for k, v in config.items() if k != 'name'}
        env = LineMsgEnv(num_agents=5, **env_config, render_mode="human")
        
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Neighborhood size: {env.n_obs_nghbr}")
        print(f"  Observation dimension per agent: {obs.shape[1]}")
        
        # Take a few steps
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
        
        env.close()
    
    print("Observation neighborhood tests completed!\n")


def test_visualization():
    """Test environment visualization."""
    print("=== Testing Visualization ===")
    
    # Create environment with rgb_array rendering
    env = LineMsgEnv(
        num_agents=8, 
        render_mode="rgb_array"
    )
    
    obs, info = env.reset()
    
    # Take a few steps and visualize
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        render_output = env.render()
        if render_output is not None:
            if isinstance(render_output, np.ndarray):
                print(f"Step {step}: Render output shape = {render_output.shape}")
            else:
                print(f"Step {step}: Render output type = {type(render_output)}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Visualization test completed!\n")


def main():
    """Run all tests."""
    print("Line Message Environment Test Suite")
    print("=" * 40)
    
    try:
        test_basic_usage()
        test_different_configurations()
        test_message_passing()
        test_neighborhood_observation()
        test_visualization()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 