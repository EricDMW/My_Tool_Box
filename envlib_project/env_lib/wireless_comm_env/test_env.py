"""
Test script for the new Wireless Communication environment.

This script demonstrates how to use the new wireless communication environment with
different configurations and parameters.
"""

import numpy as np
import gymnasium as gym
from wireless_comm_env import WirelessCommEnv


def test_basic_usage():
    """Test basic environment usage."""
    print("=== Testing Basic Environment Usage ===")
    
    # Create environment with default parameters
    env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="human")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Number of agents: {env.n_agents}")
    print(f"Grid size: {env.grid_x} x {env.grid_y}")
    
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
        {"grid_x": 2, "grid_y": 2, "name": "2x2 grid"},
        {"grid_x": 4, "grid_y": 4, "name": "4x4 grid"},
        {"grid_x": 3, "grid_y": 3, "ddl": 3, "name": "3x3 grid with ddl=3"},
        {"grid_x": 3, "grid_y": 3, "packet_arrival_probability": 0.5, "name": "3x3 grid with low arrival rate"},
    ]
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        # Remove 'name' from config for environment creation
        env_config = {k: v for k, v in config.items() if k != 'name'}
        env = WirelessCommEnv(**env_config, render_mode="human")
        
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        print(f"  Number of agents: {env.n_agents}")
        
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


def test_network_parameters():
    """Test different network parameters."""
    print("=== Testing Network Parameters ===")
    
    network_configs = [
        {"packet_arrival_probability": 0.3, "success_transmission_probability": 0.9, "name": "Low arrival, high success"},
        {"packet_arrival_probability": 0.8, "success_transmission_probability": 0.5, "name": "High arrival, low success"},
        {"packet_arrival_probability": 0.5, "success_transmission_probability": 0.7, "name": "Medium arrival, medium success"},
    ]
    
    for config in network_configs:
        print(f"\nTesting: {config['name']}")
        
        # Remove 'name' from config for environment creation
        env_config = {k: v for k, v in config.items() if k != 'name'}
        env = WirelessCommEnv(grid_x=3, grid_y=3, **env_config, render_mode="human")
        
        obs, info = env.reset()
        print(f"  Packet arrival probability: {env.p}")
        print(f"  Success transmission probability: {env.q}")
        
        # Track successful transmissions
        successful_transmissions = 0
        total_steps = 0
        
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            successful_transmissions += reward
            total_steps += 1
            
            if terminated or truncated:
                obs, info = env.reset()
        
        success_rate = successful_transmissions / total_steps if total_steps > 0 else 0
        print(f"  Success rate: {success_rate:.3f} ({successful_transmissions}/{total_steps})")
        env.close()
    
    print("Network parameter tests completed!\n")


def test_observation_neighborhood():
    """Test different observation neighborhood sizes."""
    print("=== Testing Observation Neighborhood ===")
    
    neighborhood_configs = [
        {"n_obs_neighbors": 0, "name": "No neighbors (local only)"},
        {"n_obs_neighbors": 1, "name": "1-hop neighbors"},
        {"n_obs_neighbors": 2, "name": "2-hop neighbors"},
    ]
    
    for config in neighborhood_configs:
        print(f"\nTesting: {config['name']}")
        
        # Remove 'name' from config for environment creation
        env_config = {k: v for k, v in config.items() if k != 'name'}
        env = WirelessCommEnv(grid_x=3, grid_y=3, **env_config, render_mode="human")
        
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
    env = WirelessCommEnv(
        grid_x=4, 
        grid_y=4, 
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
    print("Wireless Communication Environment Test Suite")
    print("=" * 50)
    
    try:
        test_basic_usage()
        test_different_configurations()
        test_network_parameters()
        test_observation_neighborhood()
        test_visualization()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 