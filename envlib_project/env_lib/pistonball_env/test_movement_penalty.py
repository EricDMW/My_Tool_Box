#!/usr/bin/env python3
"""
Test script for the enhanced Pistonball environment with movement penalties.
This script demonstrates the new features including:
- Multi-piston simultaneous control
- Movement penalty functionality
- Different action patterns
"""

import numpy as np
import time
from pistonball_env import PistonballEnv

def test_movement_penalty():
    """Test the movement penalty functionality."""
    print("Testing movement penalty functionality...")
    
    # Test with movement penalty
    env_with_penalty = PistonballEnv(
        n_pistons=5,
        movement_penalty=-0.1,
        movement_penalty_threshold=0.01,
        render_mode="human"
    )
    
    # Test without movement penalty
    env_no_penalty = PistonballEnv(
        n_pistons=5,
        movement_penalty=0.0,
        render_mode=None
    )
    
    obs1, info1 = env_with_penalty.reset()
    obs2, info2 = env_no_penalty.reset()
    
    total_reward_with_penalty = 0
    total_reward_no_penalty = 0
    
    # Run for a few steps with same actions
    for step in range(50):
        # Create same action for both environments
        action = np.random.uniform(-1, 1, 5)
        
        # Step with penalty
        obs1, reward1, terminated1, truncated1, info1 = env_with_penalty.step(action)
        total_reward_with_penalty += reward1
        
        # Step without penalty
        obs2, reward2, terminated2, truncated2, info2 = env_no_penalty.step(action)
        total_reward_no_penalty += reward2
        
        if terminated1 or truncated1:
            obs1, info1 = env_with_penalty.reset()
        if terminated2 or truncated2:
            obs2, info2 = env_no_penalty.reset()
    
    print(f"Total reward with penalty: {total_reward_with_penalty:.3f}")
    print(f"Total reward without penalty: {total_reward_no_penalty:.3f}")
    print(f"Difference: {total_reward_with_penalty - total_reward_no_penalty:.3f}")
    
    env_with_penalty.close()
    env_no_penalty.close()

def test_action_patterns():
    """Test different action patterns for multi-piston control."""
    print("\nTesting different action patterns...")
    
    env = PistonballEnv(n_pistons=8, render_mode="human")
    obs, info = env.reset()
    
    patterns = [
        ("All up", np.ones(8)),
        ("All down", -np.ones(8)),
        ("Stay", np.zeros(8)),
        ("Alternating", np.array([1, -1, 1, -1, 1, -1, 1, -1])),
        ("Wave", np.array([1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5])),
        ("Random", np.random.uniform(-1, 1, 8))
    ]
    
    for pattern_name, action in patterns:
        print(f"Testing pattern: {pattern_name}")
        print(f"Action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}")
        
        time.sleep(1)  # Pause to see the effect
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

def test_action_validation():
    """Test action validation and error handling."""
    print("\nTesting action validation...")
    
    env = PistonballEnv(n_pistons=5)
    obs, info = env.reset()
    
    # Test correct action shape
    try:
        action = np.ones(5)
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Correct action shape works")
    except Exception as e:
        print(f"✗ Correct action shape failed: {e}")
    
    # Test incorrect action shape
    try:
        action = np.ones(3)  # Wrong shape
        obs, reward, terminated, truncated, info = env.step(action)
        print("✗ Incorrect action shape should have failed")
    except ValueError as e:
        print(f"✓ Correctly caught incorrect action shape: {e}")
    
    # Test action clipping
    try:
        action = np.array([2.0, -2.0, 1.5, -1.5, 0.5])  # Out of range
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Action clipping works")
    except Exception as e:
        print(f"✗ Action clipping failed: {e}")
    
    env.close()

def test_different_configurations():
    """Test different environment configurations."""
    print("\nTesting different configurations...")
    
    configs = [
        {"n_pistons": 3, "movement_penalty": -0.05},
        {"n_pistons": 10, "movement_penalty": -0.1, "movement_penalty_threshold": 0.02},
        {"n_pistons": 15, "movement_penalty": -0.2, "ball_mass": 1.5},
        {"n_pistons": 5, "continuous": False, "movement_penalty": -0.1}
    ]
    
    for i, config in enumerate(configs):
        print(f"Testing configuration {i+1}: {config}")
        
        env = PistonballEnv(**config, render_mode=None)
        obs, info = env.reset()
        
        # Test action space
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        
        # Run a few steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                obs, info = env.reset()
        
        print(f"  Total reward over 10 steps: {total_reward:.3f}")
        env.close()

if __name__ == "__main__":
    print("Enhanced Pistonball Environment Test")
    print("=" * 50)
    
    try:
        test_action_validation()
        test_different_configurations()
        test_movement_penalty()
        test_action_patterns()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc() 