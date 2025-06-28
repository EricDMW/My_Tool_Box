"""
Test script for the new Pistonball environment.

This script demonstrates how to use the new pistonball environment with
different configurations and parameters.
"""
import pygame

import numpy as np
import gymnasium as gym

from pistonball_env import PistonballEnv
from manual_policy import ManualPolicy


def test_basic_usage():
    """Test basic environment usage."""
    print("=== Testing Basic Environment Usage ===")
    
    # Create environment with default parameters
    env = PistonballEnv(n_pistons=5, render_mode="rgb_array")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Number of pistons: {env.n_pistons}")
    
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
        {"n_pistons": 3, "continuous": True, "name": "3 pistons, continuous"},
        {"n_pistons": 5, "continuous": False, "name": "5 pistons, discrete"},
        {"n_pistons": 10, "ball_mass": 1.5, "name": "10 pistons, heavy ball"},
        {"n_pistons": 8, "ball_elasticity": 2.0, "name": "8 pistons, bouncy ball"},
    ]
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        # Remove 'name' from config for environment creation
        env_config = {k: v for k, v in config.items() if k != 'name'}
        env = PistonballEnv(**env_config, render_mode="rgb_array")
        
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        
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


def test_manual_control():
    """Test manual control of the environment."""
    print("=== Testing Manual Control ===")
    print("This will open a window where you can control the pistons manually.")
    print("Controls:")
    print("- W/S: Move selected piston up/down")
    print("- A/D: Select previous/next piston")
    print("- ESC: Exit")
    print("- BACKSPACE: Reset environment")
    print("\nPress Enter to start manual control (or 'n' to skip)...")
    
    user_input = input().strip().lower()
    if user_input == 'n':
        print("Skipping manual control test.\n")
        return
    
    try:
        # Create environment with human rendering
        env = PistonballEnv(
            n_pistons=8,  # Fewer pistons for easier control
            render_mode="human",
            continuous=True
        )
        
        # Create manual policy
        manual_policy = ManualPolicy(env)
        
        # Reset environment
        obs, info = env.reset()
        
        # Main loop
        clock = pygame.time.Clock()
        step_count = 0
        
        while step_count < 1000:  # Limit to 1000 steps
            clock.tick(env.metadata["render_fps"])
            
            # Get action from manual policy
            action = manual_policy(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render
            env.render()
            
            # Check if episode is done
            if terminated or truncated:
                print(f"Episode finished! Reward: {reward:.3f}")
                obs, info = env.reset()
            
            step_count += 1
        
        env.close()
        print("Manual control test completed!\n")
        
    except Exception as e:
        print(f"Error during manual control test: {e}")
        print("Skipping manual control test.\n")


def test_physics_parameters():
    """Test different physics parameters."""
    print("=== Testing Physics Parameters ===")
    
    physics_configs = [
        {"ball_mass": 0.5, "ball_friction": 0.1, "ball_elasticity": 1.0, "name": "Light, slippery, low bounce"},
        {"ball_mass": 1.0, "ball_friction": 0.5, "ball_elasticity": 1.5, "name": "Medium, normal friction, bouncy"},
        {"ball_mass": 2.0, "ball_friction": 0.8, "ball_elasticity": 0.5, "name": "Heavy, high friction, low bounce"},
    ]
    
    for config in physics_configs:
        print(f"\nTesting: {config['name']}")
        
        # Remove 'name' from config for environment creation
        env_config = {k: v for k, v in config.items() if k != 'name'}
        env = PistonballEnv(n_pistons=5, **env_config, render_mode="rgb_array")
        
        obs, info = env.reset()
        
        # Track ball movement
        if env.ball is not None:
            initial_x = env.ball.position[0]
            total_movement = 0
            
            for step in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if env.ball is not None:
                    current_x = env.ball.position[0]
                    total_movement += abs(current_x - initial_x)
                    initial_x = current_x
                
                if terminated or truncated:
                    break
            
            print(f"  Total ball movement: {total_movement:.2f} pixels")
        else:
            print("  Ball not initialized properly")
        
        env.close()
    
    print("Physics parameter tests completed!\n")


def main():
    """Run all tests."""
    print("Pistonball Environment Test Suite")
    print("=" * 40)
    
    try:
        test_basic_usage()
        test_different_configurations()
        test_physics_parameters()
        test_manual_control()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 