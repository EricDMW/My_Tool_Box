#!/usr/bin/env python3
"""
Demonstration of Enhanced Pistonball Environment Features

This script demonstrates the key enhancements made to the pistonball environment:
1. Multi-piston simultaneous control with sequence-based actions
2. Configurable movement penalties
3. Improved action validation and error handling
"""

import numpy as np
import time
from pistonball_env import PistonballEnv

def demo_multi_piston_control():
    """Demonstrate multi-piston control with sequence-based actions."""
    print("🎮 Multi-Piston Control Demonstration")
    print("=" * 50)
    
    # Create environment with 8 pistons
    env = PistonballEnv(n_pistons=8, render_mode="human")
    obs, info = env.reset()
    
    print(f"Environment created with {env.n_pistons} pistons")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Demonstrate different control patterns
    patterns = [
        ("🚀 All pistons UP", np.ones(8)),
        ("⬇️ All pistons DOWN", -np.ones(8)),
        ("⏸️ Stay STILL", np.zeros(8)),
        ("🔄 Alternating pattern", np.array([1, -1, 1, -1, 1, -1, 1, -1])),
        ("🌊 Wave pattern", np.array([1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5])),
        ("🎲 Random pattern", np.random.uniform(-1, 1, 8))
    ]
    
    for pattern_name, action in patterns:
        print(f"\n{pattern_name}")
        print(f"Action sequence: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}")
        
        # Pause to see the effect
        time.sleep(2)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("\n✅ Multi-piston control demonstration completed!")

def demo_movement_penalty():
    """Demonstrate movement penalty functionality."""
    print("\n💰 Movement Penalty Demonstration")
    print("=" * 50)
    
    # Create environments with different penalty settings
    env_no_penalty = PistonballEnv(
        n_pistons=5,
        movement_penalty=0.0,
        render_mode=None
    )
    
    env_light_penalty = PistonballEnv(
        n_pistons=5,
        movement_penalty=-0.05,
        movement_penalty_threshold=0.01,
        render_mode=None
    )
    
    env_heavy_penalty = PistonballEnv(
        n_pistons=5,
        movement_penalty=-0.2,
        movement_penalty_threshold=0.01,
        render_mode=None
    )
    
    print("Testing three environments:")
    print(f"1. No penalty: {env_no_penalty.movement_penalty}")
    print(f"2. Light penalty: {env_light_penalty.movement_penalty}")
    print(f"3. Heavy penalty: {env_heavy_penalty.movement_penalty}")
    
    # Reset all environments
    obs1, info1 = env_no_penalty.reset()
    obs2, info2 = env_light_penalty.reset()
    obs3, info3 = env_heavy_penalty.reset()
    
    # Run with same actions
    total_rewards = [0, 0, 0]
    envs = [env_no_penalty, env_light_penalty, env_heavy_penalty]
    obs_list = [obs1, obs2, obs3]
    info_list = [info1, info2, info3]
    
    print("\nRunning 50 steps with same actions...")
    for step in range(50):
        # Generate same action for all environments
        action = np.random.uniform(-1, 1, 5)
        
        for i, env in enumerate(envs):
            obs_list[i], reward, terminated, truncated, info_list[i] = env.step(action)
            total_rewards[i] += reward
            
            if terminated or truncated:
                obs_list[i], info_list[i] = env.reset()
        
        if step % 10 == 0:
            print(f"Step {step:2d}: No penalty={total_rewards[0]:6.2f}, "
                  f"Light={total_rewards[1]:6.2f}, Heavy={total_rewards[2]:6.2f}")
    
    print(f"\nFinal Results:")
    print(f"No penalty:     {total_rewards[0]:6.2f}")
    print(f"Light penalty:  {total_rewards[1]:6.2f}")
    print(f"Heavy penalty:  {total_rewards[2]:6.2f}")
    
    # Close environments
    for env in envs:
        env.close()
    
    print("✅ Movement penalty demonstration completed!")

def demo_action_validation():
    """Demonstrate action validation and error handling."""
    print("\n🔍 Action Validation Demonstration")
    print("=" * 50)
    
    env = PistonballEnv(n_pistons=5)
    obs, info = env.reset()
    
    print("Testing action validation...")
    
    # Test 1: Correct action shape
    try:
        action = np.ones(5)
        obs, reward, terminated, truncated, info = env.step(action)
        print("✅ Correct action shape (5,) works")
    except Exception as e:
        print(f"❌ Correct action shape failed: {e}")
    
    # Test 2: Incorrect action shape
    try:
        action = np.ones(3)  # Wrong shape
        obs, reward, terminated, truncated, info = env.step(action)
        print("❌ Incorrect action shape should have failed")
    except ValueError as e:
        print(f"✅ Correctly caught incorrect action shape: {e}")
    
    # Test 3: Action clipping
    try:
        action = np.array([2.0, -2.0, 1.5, -1.5, 0.5])  # Out of range
        obs, reward, terminated, truncated, info = env.step(action)
        print("✅ Action clipping works (out-of-range values clipped)")
    except Exception as e:
        print(f"❌ Action clipping failed: {e}")
    
    # Test 4: Different data types
    try:
        action = [1.0, -1.0, 0.0, 0.5, -0.5]  # List instead of numpy array
        obs, reward, terminated, truncated, info = env.step(action)
        print("✅ List action converted to numpy array")
    except Exception as e:
        print(f"❌ List action failed: {e}")
    
    env.close()
    print("✅ Action validation demonstration completed!")

def demo_different_configurations():
    """Demonstrate different environment configurations."""
    print("\n⚙️ Configuration Demonstration")
    print("=" * 50)
    
    configs = [
        {
            "name": "Small team (3 pistons)",
            "params": {"n_pistons": 3, "movement_penalty": -0.05}
        },
        {
            "name": "Medium team (10 pistons)",
            "params": {"n_pistons": 10, "movement_penalty": -0.1, "movement_penalty_threshold": 0.02}
        },
        {
            "name": "Large team (15 pistons)",
            "params": {"n_pistons": 15, "movement_penalty": -0.2, "ball_mass": 1.5}
        },
        {
            "name": "Discrete actions",
            "params": {"n_pistons": 5, "continuous": False, "movement_penalty": -0.1}
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"Parameters: {config['params']}")
        
        env = PistonballEnv(**config, render_mode=None)
        obs, info = env.reset()
        
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Run a few steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                obs, info = env.reset()
        
        print(f"Total reward over 10 steps: {total_reward:.3f}")
        env.close()
    
    print("✅ Configuration demonstration completed!")

def main():
    """Run all demonstrations."""
    print("🎯 Enhanced Pistonball Environment Demo")
    print("=" * 60)
    print("This demo showcases the enhanced features:")
    print("• Multi-piston simultaneous control")
    print("• Configurable movement penalties")
    print("• Improved action validation")
    print("• Flexible configuration options")
    print("=" * 60)
    
    try:
        demo_action_validation()
        demo_different_configurations()
        demo_movement_penalty()
        demo_multi_piston_control()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Sequence-based actions for multiple pistons")
        print("✅ Configurable movement penalties")
        print("✅ Action validation and error handling")
        print("✅ Flexible environment configuration")
        print("✅ Both continuous and discrete action spaces")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 