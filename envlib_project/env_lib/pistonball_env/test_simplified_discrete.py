#!/usr/bin/env python3
"""
Test script for the simplified discrete action space in the pistonball environment.

This script verifies that the discrete action space now uses a sequence of discrete actions
for each piston instead of the complex base-3 decoding.
"""

import numpy as np
from pistonball_env import PistonballEnv

def test_discrete_action_space():
    """Test the simplified discrete action space."""
    print("Testing simplified discrete action space...")
    
    # Create discrete environment
    env = PistonballEnv(n_pistons=5, continuous=False)
    obs, info = env.reset()
    
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action space low: {env.action_space.low}")
    print(f"Action space high: {env.action_space.high}")
    
    # Test specific discrete actions
    test_actions = [
        np.array([0, 0, 0, 0, 0]),  # All down
        np.array([1, 1, 1, 1, 1]),  # All stay
        np.array([2, 2, 2, 2, 2]),  # All up
        np.array([0, 1, 2, 1, 0]),  # down, stay, up, stay, down
        np.array([2, 1, 0, 1, 2]),  # up, stay, down, stay, up
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\nTest action {i+1}: {action}")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # Test random sampling
    print(f"\nTesting random action sampling:")
    for i in range(5):
        action = env.action_space.sample()
        print(f"Random action {i+1}: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("✅ Discrete action space test completed!")

def test_discrete_vs_continuous():
    """Compare discrete and continuous action spaces."""
    print("\nComparing discrete vs continuous action spaces...")
    
    # Create both environments
    env_discrete = PistonballEnv(n_pistons=3, continuous=False)
    env_continuous = PistonballEnv(n_pistons=3, continuous=True)
    
    print(f"Discrete action space: {env_discrete.action_space}")
    print(f"Continuous action space: {env_continuous.action_space}")
    
    # Test equivalent actions
    discrete_action = np.array([0, 1, 2])  # down, stay, up
    continuous_action = np.array([-1, 0, 1])  # down, stay, up
    
    print(f"\nDiscrete action: {discrete_action}")
    print(f"Continuous action: {continuous_action}")
    
    # Reset both environments
    obs_d, info_d = env_discrete.reset()
    obs_c, info_c = env_continuous.reset()
    
    # Take steps
    obs_d, reward_d, terminated_d, truncated_d, info_d = env_discrete.step(discrete_action)
    obs_c, reward_c, terminated_c, truncated_c, info_c = env_continuous.step(continuous_action)
    
    print(f"Discrete reward: {reward_d:.3f}")
    print(f"Continuous reward: {reward_c:.3f}")
    print(f"Rewards match: {abs(reward_d - reward_c) < 0.001}")
    
    env_discrete.close()
    env_continuous.close()
    print("✅ Discrete vs continuous comparison completed!")

def test_action_validation_discrete():
    """Test action validation for discrete actions."""
    print("\nTesting discrete action validation...")
    
    env = PistonballEnv(n_pistons=4, continuous=False)
    obs, info = env.reset()
    
    # Test valid actions
    valid_actions = [
        np.array([0, 1, 2, 0]),
        np.array([1, 1, 1, 1]),
        np.array([2, 0, 1, 2]),
    ]
    
    for action in valid_actions:
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✅ Valid action {action} works")
        except Exception as e:
            print(f"❌ Valid action {action} failed: {e}")
    
    # Test invalid actions
    invalid_actions = [
        np.array([0, 1, 2]),  # Wrong shape
        np.array([0, 1, 2, 3]),  # Out of range
        np.array([0, 1, 2, -1]),  # Out of range
    ]
    
    for action in invalid_actions:
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"❌ Invalid action {action} should have failed")
        except Exception as e:
            print(f"✅ Correctly caught invalid action {action}: {e}")
    
    env.close()
    print("✅ Discrete action validation test completed!")

def test_different_piston_counts():
    """Test discrete actions with different piston counts."""
    print("\nTesting discrete actions with different piston counts...")
    
    piston_counts = [1, 2, 3, 5, 8]
    
    for n_pistons in piston_counts:
        print(f"\nTesting {n_pistons} pistons:")
        
        env = PistonballEnv(n_pistons=n_pistons, continuous=False)
        obs, info = env.reset()
        
        print(f"  Action space: {env.action_space}")
        print(f"  Action space shape: {env.action_space.shape}")
        
        # Test action sampling
        action = env.action_space.sample()
        print(f"  Sampled action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Reward: {reward:.3f}")
        
        env.close()
    
    print("✅ Different piston counts test completed!")

def main():
    """Run all tests."""
    print("🎯 Simplified Discrete Action Space Test")
    print("=" * 60)
    print("This test verifies the simplified discrete action space implementation.")
    print("Discrete actions are now sequences of discrete values (0, 1, 2) for each piston.")
    
    try:
        test_discrete_action_space()
        test_discrete_vs_continuous()
        test_action_validation_discrete()
        test_different_piston_counts()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("\nKey Changes:")
        print("• Discrete action space is now Box(0, 2, shape=(n_pistons,))")
        print("• No more complex base-3 decoding")
        print("• Direct sequence of discrete actions for each piston")
        print("• 0=down, 1=stay, 2=up for each piston")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 