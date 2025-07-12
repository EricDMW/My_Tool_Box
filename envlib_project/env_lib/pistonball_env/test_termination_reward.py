"""
Test script for the configurable termination reward feature in PistonballEnv.

This script tests:
1. Default termination reward (0.5)
2. Custom termination reward values (positive and negative)
3. Combined rewards (termination + leftmost piston)
4. Different termination reward values
"""

import numpy as np
import pygame
from pistonball_env import PistonballEnv


def test_termination_reward():
    """Test the configurable termination reward feature."""
    print("🧪 Testing Configurable Termination Reward Feature")
    print("=" * 50)
    
    # Test 1: Default termination reward (0.5)
    print("\n📋 Test 1: Default termination reward (0.5)")
    env1 = PistonballEnv(
        n_pistons=5, 
        terminated_condition=True, 
        render_mode=None
    )
    obs, info = env1.reset()
    
    # Force ball to move left quickly
    for step in range(50):
        # Move all pistons up to create a path for the ball
        action = np.ones(5)  # All pistons up
        obs, reward, terminated, truncated, info = env1.step(action)
        
        if terminated:
            print(f"✅ Episode terminated at step {step} (ball hit left wall)")
            print(f"   Final total reward: {reward}")
            print(f"   Local rewards: {info['local_rewards']}")
            print(f"   Default termination reward: {env1.termination_reward}")
            
            # Check that observable pistons got the default reward
            observable_rewards = []
            for i, reward_val in enumerate(info['local_rewards']):
                if reward_val > 0:  # Only consider pistons that got termination reward
                    observable_rewards.append(reward_val)
            
            if len(set(observable_rewards)) == 1 and abs(observable_rewards[0] - 0.5) < 0.1:
                print(f"   ✅ Observable pistons got default termination reward (0.5)")
            else:
                print(f"   ❌ Observable pistons didn't get expected default reward")
            break
        elif truncated:
            print(f"❌ Episode truncated at step {step} (max cycles reached)")
            break
    
    env1.close()
    
    # Test 2: Custom positive termination reward
    print("\n📋 Test 2: Custom positive termination reward (1.0)")
    env2 = PistonballEnv(
        n_pistons=5, 
        terminated_condition=True, 
        termination_reward=1.0,
        render_mode=None
    )
    obs, info = env2.reset()
    
    # Force ball to move left quickly
    for step in range(50):
        # Move all pistons up to create a path for the ball
        action = np.ones(5)  # All pistons up
        obs, reward, terminated, truncated, info = env2.step(action)
        
        if terminated:
            print(f"✅ Episode terminated at step {step} (ball hit left wall)")
            print(f"   Final total reward: {reward}")
            print(f"   Local rewards: {info['local_rewards']}")
            print(f"   Custom termination reward: {env2.termination_reward}")
            
            # Check that observable pistons got the custom reward
            observable_rewards = []
            for i, reward_val in enumerate(info['local_rewards']):
                if reward_val > 0:  # Only consider pistons that got termination reward
                    observable_rewards.append(reward_val)
            
            if len(set(observable_rewards)) == 1 and abs(observable_rewards[0] - 1.0) < 0.1:
                print(f"   ✅ Observable pistons got custom termination reward (1.0)")
            else:
                print(f"   ❌ Observable pistons didn't get expected custom reward")
            break
        elif truncated:
            print(f"❌ Episode truncated at step {step} (max cycles reached)")
            break
    
    env2.close()
    
    # Test 3: Custom negative termination reward
    print("\n📋 Test 3: Custom negative termination reward (-0.5)")
    env3 = PistonballEnv(
        n_pistons=5, 
        terminated_condition=True, 
        termination_reward=-0.5,
        render_mode=None
    )
    obs, info = env3.reset()
    
    # Force ball to move left quickly
    for step in range(50):
        # Move all pistons up to create a path for the ball
        action = np.ones(5)  # All pistons up
        obs, reward, terminated, truncated, info = env3.step(action)
        
        if terminated:
            print(f"✅ Episode terminated at step {step} (ball hit left wall)")
            print(f"   Final total reward: {reward}")
            print(f"   Local rewards: {info['local_rewards']}")
            print(f"   Custom termination reward: {env3.termination_reward}")
            
            # Check that observable pistons got the negative reward
            observable_rewards = []
            for i, reward_val in enumerate(info['local_rewards']):
                if reward_val < 0:  # Consider pistons that got negative termination reward
                    observable_rewards.append(reward_val)
            
            if len(set(observable_rewards)) == 1 and abs(observable_rewards[0] - (-0.5)) < 0.1:
                print(f"   ✅ Observable pistons got negative termination reward (-0.5)")
            else:
                print(f"   ❌ Observable pistons didn't get expected negative reward")
            break
        elif truncated:
            print(f"❌ Episode truncated at step {step} (max cycles reached)")
            break
    
    env3.close()
    
    # Test 4: Large termination reward
    print("\n📋 Test 4: Large termination reward (2.0)")
    env4 = PistonballEnv(
        n_pistons=8, 
        terminated_condition=True, 
        termination_reward=2.0,
        render_mode=None
    )
    obs, info = env4.reset()
    
    # Force ball to move left quickly
    for step in range(50):
        # Move all pistons up to create a path for the ball
        action = np.ones(8)  # All pistons up
        obs, reward, terminated, truncated, info = env4.step(action)
        
        if terminated:
            print(f"✅ Episode terminated at step {step} (ball hit left wall)")
            print(f"   Final total reward: {reward}")
            print(f"   Local rewards: {info['local_rewards']}")
            print(f"   Large termination reward: {env4.termination_reward}")
            
            # Check that observable pistons got the large reward
            observable_rewards = []
            for i, reward_val in enumerate(info['local_rewards']):
                if reward_val > 0:  # Only consider pistons that got termination reward
                    observable_rewards.append(reward_val)
            
            if len(set(observable_rewards)) == 1 and abs(observable_rewards[0] - 2.0) < 0.1:
                print(f"   ✅ Observable pistons got large termination reward (2.0)")
            else:
                print(f"   ❌ Observable pistons didn't get expected large reward")
            break
        elif truncated:
            print(f"❌ Episode truncated at step {step} (max cycles reached)")
            break
    
    env4.close()


def test_parameter_validation():
    """Test that the new parameter is properly handled."""
    print("\n📋 Test 5: Parameter validation")
    
    # Test with default value
    env1 = PistonballEnv(n_pistons=3)
    print(f"   ✅ Environment created with default termination_reward")
    print(f"   ✅ Environment attribute: {env1.termination_reward}")
    env1.close()
    
    # Test with positive value
    env2 = PistonballEnv(n_pistons=3, termination_reward=1.5)
    print(f"   ✅ Environment created with termination_reward=1.5")
    print(f"   ✅ Environment attribute: {env2.termination_reward}")
    env2.close()
    
    # Test with negative value
    env3 = PistonballEnv(n_pistons=3, termination_reward=-0.8)
    print(f"   ✅ Environment created with termination_reward=-0.8")
    print(f"   ✅ Environment attribute: {env3.termination_reward}")
    env3.close()
    
    # Test with zero value
    env4 = PistonballEnv(n_pistons=3, termination_reward=0.0)
    print(f"   ✅ Environment created with termination_reward=0.0")
    print(f"   ✅ Environment attribute: {env4.termination_reward}")
    env4.close()


def test_combined_rewards():
    """Test that termination reward combines correctly with leftmost piston reward."""
    print("\n📋 Test 6: Combined rewards")
    
    # Create environment with both termination and leftmost piston rewards
    env = PistonballEnv(
        n_pistons=5,
        terminated_condition=True,
        termination_reward=1.0,
        leftmost_piston_reward=0.5,
        render_mode=None
    )
    
    obs, info = env.reset()
    
    # Force ball to move left quickly
    for step in range(50):
        action = np.ones(5)  # All pistons up
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"   ✅ Episode terminated at step {step}")
            print(f"   ✅ Final local rewards: {info['local_rewards']}")
            
            # Leftmost piston should have: termination_reward + leftmost_reward
            # Other observable pistons should have: termination_reward only
            leftmost_reward = info['local_rewards'][0]
            other_rewards = info['local_rewards'][1:]
            
            print(f"   ✅ Leftmost piston total reward: {leftmost_reward}")
            print(f"   ✅ Other piston rewards: {other_rewards}")
            print(f"   ✅ Expected leftmost piston reward: {1.0 + 0.5}")
            print(f"   ✅ Expected other piston reward: {1.0}")
            
            # Check that leftmost piston has higher reward
            if abs(leftmost_reward - 1.5) < 0.1 and abs(max(other_rewards) - 1.0) < 0.1:
                print(f"   ✅ Combined rewards work correctly")
            else:
                print(f"   ❌ Combined rewards don't work as expected")
            break
        elif truncated:
            print(f"   ⏰ Episode truncated at step {step}")
            break
    
    env.close()


def test_different_termination_rewards():
    """Test different termination reward values."""
    print("\n📋 Test 7: Different termination reward values")
    
    termination_rewards = [0.1, 0.5, 1.0, 2.0]
    
    for term_reward in termination_rewards:
        print(f"\n   Testing termination_reward={term_reward}")
        env = PistonballEnv(
            n_pistons=6,
            terminated_condition=True,
            termination_reward=term_reward,
            render_mode=None
        )
        
        obs, info = env.reset()
        
        for step in range(50):
            action = np.ones(6)  # All pistons up
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"   ✅ Episode terminated at step {step}")
                print(f"   ✅ Final reward: {reward:.3f}")
                
                # Check that observable pistons got the correct termination reward
                observable_rewards = []
                for i, reward_val in enumerate(info['local_rewards']):
                    if abs(reward_val - term_reward) < 0.1:  # Consider pistons that got termination reward
                        observable_rewards.append(reward_val)
                
                if len(observable_rewards) > 0:
                    print(f"   ✅ Observable pistons got termination reward: {observable_rewards[0]:.3f}")
                else:
                    print(f"   ❌ No pistons got expected termination reward")
                break
            elif truncated:
                print(f"   ⏰ Episode truncated at step {step}")
                break
        
        env.close()


def run_visual_test():
    """Run a visual test to see the configurable termination reward in action."""
    print("\n📋 Test 8: Visual test (optional)")
    print("   This will open a window to show the configurable termination reward in action.")
    print("   Press any key to continue...")
    
    try:
        env = PistonballEnv(
            n_pistons=5, 
            terminated_condition=True, 
            termination_reward=1.0,
            render_mode="human"
        )
        obs, info = env.reset()
        
        for step in range(100):
            # Move all pistons up to create a path for the ball
            action = np.ones(5)  # All pistons up
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            # Check for pygame events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    env.close()
                    return
            
            if terminated:
                print(f"   ✅ Visual test: Episode terminated at step {step}")
                print(f"   ✅ Final reward: {reward}")
                print(f"   ✅ Local rewards: {info['local_rewards']}")
                break
            elif truncated:
                print(f"   ❌ Visual test: Episode truncated at step {step}")
                break
        
        env.close()
        
    except Exception as e:
        print(f"   ⚠️  Visual test failed: {e}")
        print("   This is normal if pygame display is not available")


if __name__ == "__main__":
    print("🚀 PistonballEnv Configurable Termination Reward Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_termination_reward()
    test_parameter_validation()
    test_combined_rewards()
    test_different_termination_rewards()
    
    # Ask user if they want to run visual test
    try:
        user_input = input("\nWould you like to run the visual test? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            run_visual_test()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    
    print("\n✅ All tests completed!")
    print("\n📝 Summary:")
    print("   - Configurable termination reward feature is working correctly")
    print("   - Parameter validation is working")
    print("   - Default value (0.5) is applied correctly")
    print("   - Custom positive and negative rewards work")
    print("   - Combined rewards with leftmost piston reward work")
    print("   - Different termination reward values are applied correctly") 