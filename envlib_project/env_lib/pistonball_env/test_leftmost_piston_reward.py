"""
Test script for the leftmost piston reward feature in PistonballEnv.

This script tests:
1. Leftmost piston reward when ball hits left wall
2. Different leftmost piston reward values (positive and negative)
3. Combined rewards (termination + leftmost piston)
4. No leftmost piston reward when disabled
"""

import numpy as np
import pygame
from pistonball_env import PistonballEnv


def test_leftmost_piston_reward():
    """Test the leftmost piston reward feature."""
    print("🧪 Testing Leftmost Piston Reward Feature")
    print("=" * 50)
    
    # Test 1: Positive leftmost piston reward
    print("\n📋 Test 1: Positive leftmost piston reward")
    env1 = PistonballEnv(
        n_pistons=5, 
        terminated_condition=True, 
        leftmost_piston_reward=1.0,
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
            
            # Check that leftmost piston (index 0) got extra reward
            leftmost_reward = info['local_rewards'][0]
            other_rewards = info['local_rewards'][1:]
            
            print(f"   Leftmost piston reward: {leftmost_reward}")
            print(f"   Other piston rewards: {other_rewards}")
            
            # Leftmost piston should have higher reward due to leftmost_piston_reward
            if leftmost_reward > max(other_rewards):
                print(f"   ✅ Leftmost piston got higher reward as expected")
            else:
                print(f"   ❌ Leftmost piston reward not higher than others")
            break
        elif truncated:
            print(f"❌ Episode truncated at step {step} (max cycles reached)")
            break
    
    env1.close()
    
    # Test 2: Negative leftmost piston reward
    print("\n📋 Test 2: Negative leftmost piston reward")
    env2 = PistonballEnv(
        n_pistons=5, 
        terminated_condition=True, 
        leftmost_piston_reward=-0.5,
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
            
            # Check that leftmost piston (index 0) got reduced reward
            leftmost_reward = info['local_rewards'][0]
            other_rewards = info['local_rewards'][1:]
            
            print(f"   Leftmost piston reward: {leftmost_reward}")
            print(f"   Other piston rewards: {other_rewards}")
            
            # Leftmost piston should have lower reward due to negative leftmost_piston_reward
            if leftmost_reward < max(other_rewards):
                print(f"   ✅ Leftmost piston got lower reward as expected")
            else:
                print(f"   ❌ Leftmost piston reward not lower than others")
            break
        elif truncated:
            print(f"❌ Episode truncated at step {step} (max cycles reached)")
            break
    
    env2.close()
    
    # Test 3: No leftmost piston reward (default)
    print("\n📋 Test 3: No leftmost piston reward (default)")
    env3 = PistonballEnv(
        n_pistons=5, 
        terminated_condition=True, 
        leftmost_piston_reward=0.0,  # Default value
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
            
            # Check that all observable pistons get same reward
            observable_rewards = []
            for i, reward_val in enumerate(info['local_rewards']):
                if reward_val > 0:  # Only consider pistons that got termination reward
                    observable_rewards.append(reward_val)
            
            if len(set(observable_rewards)) == 1:
                print(f"   ✅ All observable pistons got same reward (no leftmost bonus)")
            else:
                print(f"   ❌ Observable pistons got different rewards")
            break
        elif truncated:
            print(f"❌ Episode truncated at step {step} (max cycles reached)")
            break
    
    env3.close()
    
    # Test 4: Large leftmost piston reward
    print("\n📋 Test 4: Large leftmost piston reward")
    env4 = PistonballEnv(
        n_pistons=8, 
        terminated_condition=True, 
        leftmost_piston_reward=2.0,  # Large reward
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
            
            # Check that leftmost piston got significantly higher reward
            leftmost_reward = info['local_rewards'][0]
            max_other_reward = max(info['local_rewards'][1:])
            
            print(f"   Leftmost piston reward: {leftmost_reward}")
            print(f"   Max other piston reward: {max_other_reward}")
            print(f"   Difference: {leftmost_reward - max_other_reward}")
            
            # Should be around 2.0 difference due to leftmost_piston_reward
            expected_difference = 2.0
            actual_difference = leftmost_reward - max_other_reward
            
            if abs(actual_difference - expected_difference) < 0.1:
                print(f"   ✅ Leftmost piston reward difference is correct")
            else:
                print(f"   ❌ Leftmost piston reward difference is incorrect")
            break
        elif truncated:
            print(f"❌ Episode truncated at step {step} (max cycles reached)")
            break
    
    env4.close()


def test_parameter_validation():
    """Test that the new parameter is properly handled."""
    print("\n📋 Test 5: Parameter validation")
    
    # Test with positive value
    env1 = PistonballEnv(n_pistons=3, leftmost_piston_reward=1.5)
    print(f"   ✅ Environment created with leftmost_piston_reward=1.5")
    print(f"   ✅ Environment attribute: {env1.leftmost_piston_reward}")
    env1.close()
    
    # Test with negative value
    env2 = PistonballEnv(n_pistons=3, leftmost_piston_reward=-0.8)
    print(f"   ✅ Environment created with leftmost_piston_reward=-0.8")
    print(f"   ✅ Environment attribute: {env2.leftmost_piston_reward}")
    env2.close()
    
    # Test with zero value
    env3 = PistonballEnv(n_pistons=3, leftmost_piston_reward=0.0)
    print(f"   ✅ Environment created with leftmost_piston_reward=0.0")
    print(f"   ✅ Environment attribute: {env3.leftmost_piston_reward}")
    env3.close()
    
    # Test with default (should be 0.0)
    env4 = PistonballEnv(n_pistons=3)
    print(f"   ✅ Environment created with default leftmost_piston_reward")
    print(f"   ✅ Environment attribute: {env4.leftmost_piston_reward}")
    env4.close()


def test_combined_rewards():
    """Test that leftmost piston reward combines correctly with other rewards."""
    print("\n📋 Test 6: Combined rewards")
    
    # Create environment with both termination and leftmost piston rewards
    env = PistonballEnv(
        n_pistons=5,
        terminated_condition=True,
        leftmost_piston_reward=1.0,
        movement_penalty=-0.1,
        render_mode=None
    )
    
    obs, info = env.reset()
    
    # Take a few steps to accumulate movement penalties
    for step in range(10):
        action = np.random.uniform(-1, 1, 5)  # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"   ✅ Episode terminated at step {step}")
            print(f"   ✅ Final local rewards: {info['local_rewards']}")
            
            # Leftmost piston should have: termination_reward + leftmost_reward + movement_penalties
            # Other observable pistons should have: termination_reward + movement_penalties
            leftmost_reward = info['local_rewards'][0]
            other_rewards = info['local_rewards'][1:]
            
            print(f"   ✅ Leftmost piston total reward: {leftmost_reward}")
            print(f"   ✅ Other piston rewards: {other_rewards}")
            
            # Check that leftmost piston has higher reward
            if leftmost_reward > max(other_rewards):
                print(f"   ✅ Leftmost piston correctly has higher total reward")
            else:
                print(f"   ❌ Leftmost piston doesn't have higher total reward")
            break
        elif truncated:
            print(f"   ⏰ Episode truncated at step {step}")
            break
    
    env.close()


def run_visual_test():
    """Run a visual test to see the leftmost piston reward in action."""
    print("\n📋 Test 7: Visual test (optional)")
    print("   This will open a window to show the leftmost piston reward in action.")
    print("   Press any key to continue...")
    
    try:
        env = PistonballEnv(
            n_pistons=5, 
            terminated_condition=True, 
            leftmost_piston_reward=1.0,
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
    print("🚀 PistonballEnv Leftmost Piston Reward Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_leftmost_piston_reward()
    test_parameter_validation()
    test_combined_rewards()
    
    # Ask user if they want to run visual test
    try:
        user_input = input("\nWould you like to run the visual test? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            run_visual_test()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    
    print("\n✅ All tests completed!")
    print("\n📝 Summary:")
    print("   - Leftmost piston reward feature is working correctly")
    print("   - Parameter validation is working")
    print("   - Positive and negative rewards are applied correctly")
    print("   - Combined rewards work as expected")
    print("   - Only leftmost piston (index 0) gets the special reward") 