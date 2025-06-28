"""
Simple example demonstrating the Wireless Communication environment.

This script shows how to create and use the wireless communication environment
with different configurations.
"""

import numpy as np
from wireless_comm_env import WirelessCommEnv


def basic_example():
    """Basic example of using the wireless communication environment."""
    print("=== Basic Wireless Communication Example ===")
    
    # Create environment with 3x3 grid
    env = WirelessCommEnv(
        grid_x=3,
        grid_y=3,
        render_mode="human"
    )
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Number of agents: {env.n_agents}")
    print(f"Grid size: {env.grid_x} x {env.grid_y}")
    
    # Run for 20 steps
    total_reward = 0
    for step in range(20):
        # Take random actions
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
        
        # Reset if episode is done
        if terminated or truncated:
            print(f"Episode finished at step {step + 1}")
            obs, info = env.reset()
            total_reward = 0
    
    env.close()
    print("Basic example completed!\n")


def network_parameters_example():
    """Example with different network parameters."""
    print("=== Network Parameters Example ===")
    
    # Create environment with different network characteristics
    env = WirelessCommEnv(
        grid_x=4,
        grid_y=4,
        packet_arrival_probability=0.6,      # Medium packet arrival rate
        success_transmission_probability=0.8, # High success rate
        render_mode="human"
    )
    
    obs, info = env.reset()
    print(f"Environment created with network parameters:")
    print(f"Packet arrival probability: {env.p}")
    print(f"Success transmission probability: {env.q}")
    
    # Run for 30 steps
    successful_transmissions = 0
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        successful_transmissions += reward
        
        if step % 10 == 0:
            print(f"Step {step}: Successful transmissions so far = {successful_transmissions}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    success_rate = successful_transmissions / 30
    print(f"Final success rate: {success_rate:.3f} ({successful_transmissions}/30)")
    env.close()
    print("Network parameters example completed!\n")


def deadline_example():
    """Example with different deadline horizons."""
    print("=== Deadline Horizon Example ===")
    
    # Create environment with longer deadline horizon
    env = WirelessCommEnv(
        grid_x=3,
        grid_y=3,
        ddl=4,  # Longer deadline horizon
        render_mode="human"
    )
    
    obs, info = env.reset()
    print(f"Environment created with deadline horizon: {env.ddl}")
    print(f"Observation shape: {obs.shape}")
    
    # Run for 15 steps
    for step in range(15):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 5 == 0:
            print(f"Step {step}: Reward = {reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Deadline horizon example completed!\n")


def neighborhood_example():
    """Example with different observation neighborhoods."""
    print("=== Observation Neighborhood Example ===")
    
    # Create environment with larger observation neighborhood
    env = WirelessCommEnv(
        grid_x=4,
        grid_y=4,
        n_obs_neighbors=2,  # 2-hop neighborhood
        render_mode="human"
    )
    
    obs, info = env.reset()
    print(f"Environment created with observation neighborhood size: {env.n_obs_nghbr}")
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dimension per agent: {obs.shape[1]}")
    
    # Run for 10 steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 5 == 0:
            print(f"Step {step}: Reward = {reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Observation neighborhood example completed!\n")


def main():
    """Run all examples."""
    print("Wireless Communication Environment Examples")
    print("=" * 45)
    
    try:
        basic_example()
        network_parameters_example()
        deadline_example()
        neighborhood_example()
        
        print("All examples completed successfully!")
        print("\nTo test the environment more thoroughly, run:")
        print("python test_env.py")
        
    except Exception as e:
        print(f"Error during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 