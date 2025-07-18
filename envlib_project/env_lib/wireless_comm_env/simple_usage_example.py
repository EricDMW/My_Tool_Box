"""
Simple usage example for the Wireless Communication Environment.

This script demonstrates basic usage of the environment with different rendering options.
"""

import numpy as np
from wireless_comm_env import WirelessCommEnv


def basic_usage_example():
    """Demonstrate basic environment usage without rendering."""
    print("=== Basic Usage Example ===")
    
    # Create environment
    env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode=None)
    obs, info = env.reset()
    
    print(f"Environment created with {env.n_agents} agents")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    total_reward = 0
    for step in range(10):
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
            total_reward = 0
    
    env.close()
    print("Basic example completed!")


def rendering_example():
    """Demonstrate environment with rendering."""
    print("\n=== Rendering Example ===")
    
    # Create environment with rendering
    env = WirelessCommEnv(grid_x=4, grid_y=4, render_mode="rgb_array")
    obs, info = env.reset()
    
    print("Running environment with rendering...")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the current state
        frame = env.render()
        print(f"Step {step + 1}: Rendered frame shape = {frame.shape}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Rendering example completed!")


def realtime_rendering_example():
    """Demonstrate real-time rendering during training."""
    print("\n=== Real-time Rendering Example ===")
    
    # Create environment
    env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="rgb_array")
    obs, info = env.reset()
    
    print("Running with real-time rendering (showing every 2 steps)...")
    
    for step in range(8):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Show real-time rendering every 2 steps
        if step % 2 == 0:
            print(f"Step {step + 1}: Showing real-time rendering...")
            env.render_realtime(show=True)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Real-time rendering example completed!")


def gif_creation_example():
    """Demonstrate GIF creation from training."""
    print("\n=== GIF Creation Example ===")
    
    # Create environment
    env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="rgb_array")
    obs, info = env.reset()
    
    # Start collecting frames
    env.start_frame_collection()
    
    print("Running environment and collecting frames...")
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to collect frames
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # Stop collection and save GIF
    env.stop_frame_collection()
    gif_path = env.save_gif(gif_path="simple_usage_example.gif", fps=3)
    
    env.close()
    print(f"GIF saved to: {gif_path}")


def custom_parameters_example():
    """Demonstrate environment with custom parameters."""
    print("\n=== Custom Parameters Example ===")
    
    # Create environment with custom parameters
    env = WirelessCommEnv(
        grid_x=5,
        grid_y=5,
        ddl=3,
        packet_arrival_probability=0.9,
        success_transmission_probability=0.7,
        n_obs_neighbors=2,
        max_iter=30,
        render_mode="rgb_array"
    )
    
    obs, info = env.reset()
    
    print(f"Custom environment created:")
    print(f"- Grid size: {env.grid_x}x{env.grid_y}")
    print(f"- Deadline: {env.ddl}")
    print(f"- Packet arrival probability: {env.p}")
    print(f"- Success probability: {env.q}")
    print(f"- Observation neighbors: {env.n_obs_nghbr}")
    print(f"- Max iterations: {env.max_iter}")
    
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
            total_reward = 0
    
    env.close()
    print("Custom parameters example completed!")


if __name__ == "__main__":
    print("Wireless Communication Environment - Simple Usage Examples")
    print("=" * 60)
    
    # Run all examples
    basic_usage_example()
    rendering_example()
    realtime_rendering_example()
    gif_creation_example()
    custom_parameters_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Check the generated files for results.") 