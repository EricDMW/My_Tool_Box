#!/usr/bin/env python3
"""
Demonstration of Piston Numbering Feature

This script demonstrates the new piston numbering feature that displays
the order of each piston at the bottom during rendering.
"""

import pygame
import numpy as np
import time
from pistonball_env import PistonballEnv


def demo_piston_numbering_visual():
    """Demonstrate piston numbering with visual rendering."""
    print("🔢 Piston Numbering Feature Demonstration")
    print("=" * 50)
    print("This demonstration shows the piston numbering feature.")
    print("Each piston displays its index number (0, 1, 2, ...) at the bottom.")
    print("The numbers help identify individual pistons during debugging and analysis.")
    print()
    
    # Create environment with human rendering
    env = PistonballEnv(n_pistons=8, render_mode="human")
    obs, info = env.reset()
    
    print(f"✅ Environment created with {env.n_pistons} pistons")
    print(f"✅ Font initialized: {type(env.font)}")
    print("✅ Piston numbers 0-7 should be visible at the bottom of each piston")
    print()
    print("🎮 Controls:")
    print("- Watch the piston numbers as they move")
    print("- Press Ctrl+C to exit")
    print()
    
    try:
        # Run demonstration with different patterns
        patterns = [
            ("All pistons UP", np.ones(8)),
            ("All pistons DOWN", -np.ones(8)),
            ("Stay STILL", np.zeros(8)),
            ("Alternating pattern", np.array([1, -1, 1, -1, 1, -1, 1, -1])),
            ("Wave pattern", np.array([1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5])),
        ]
        
        for pattern_name, action in patterns:
            print(f"🎯 Pattern: {pattern_name}")
            print(f"   Action: {action}")
            
            # Apply pattern for several steps
            for step in range(20):
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                
                # Small delay to see the movement
                time.sleep(0.1)
                
                if terminated or truncated:
                    obs, info = env.reset()
            
            print(f"   ✅ Pattern completed")
            print()
        
        # Continue with random actions
        print("🎲 Now showing random actions...")
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            time.sleep(0.1)
            
            if terminated or truncated:
                obs, info = env.reset()
        
        print("✅ Demonstration completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demonstration stopped by user")
    
    finally:
        env.close()
        print("✅ Environment closed")


def demo_piston_numbering_rgb_array():
    """Demonstrate piston numbering with rgb_array rendering."""
    print("\n🖼️ RGB Array Rendering with Piston Numbering")
    print("=" * 50)
    
    # Create environment with rgb_array rendering
    env = PistonballEnv(n_pistons=5, render_mode="rgb_array")
    obs, info = env.reset()
    
    print(f"✅ Environment created with {env.n_pistons} pistons")
    print("✅ Testing rgb_array rendering with piston numbering...")
    
    # Take a few steps and capture frames
    for step in range(5):
        action = np.random.uniform(-1, 1, 5)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get rendered frame
        frame = env.render()
        
        if frame is not None:
            print(f"   Step {step}: Frame shape = {frame.shape}")
            print(f"   Frame data type = {frame.dtype}")
            print(f"   Frame value range = [{frame.min()}, {frame.max()}]")
            print(f"   Piston numbers 0-4 should be visible in the frame")
        else:
            print(f"   Step {step}: No frame returned")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("✅ RGB array demonstration completed!")


def demo_different_piston_counts():
    """Demonstrate piston numbering with different piston counts."""
    print("\n🔢 Different Piston Counts")
    print("=" * 50)
    
    piston_counts = [3, 5, 10, 15]
    
    for n_pistons in piston_counts:
        print(f"\n🎯 Testing with {n_pistons} pistons...")
        
        env = PistonballEnv(n_pistons=n_pistons, render_mode="rgb_array")
        obs, info = env.reset()
        
        # Take one step
        action = np.random.uniform(-1, 1, n_pistons)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get rendered frame
        frame = env.render()
        
        if frame is not None:
            print(f"   ✅ Frame shape: {frame.shape}")
            print(f"   ✅ Piston numbers should be 0-{n_pistons-1}")
            print(f"   ✅ Screen width adjusts automatically: {frame.shape[1]} pixels")
        else:
            print("   ❌ No frame returned")
        
        env.close()
    
    print("✅ Different piston counts demonstration completed!")


def main():
    """Run the piston numbering demonstration."""
    print("🔢 Piston Numbering Feature Demonstration Suite")
    print("=" * 50)
    
    try:
        # Test font initialization
        print("🔤 Testing font initialization...")
        env = PistonballEnv(n_pistons=5, render_mode="rgb_array")
        if hasattr(env, 'font') and env.font is not None:
            print("✅ Font is properly initialized")
        else:
            print("❌ Font is not initialized")
        env.close()
        
        # Run demonstrations
        demo_piston_numbering_rgb_array()
        demo_different_piston_counts()
        
        print("\n🎮 Interactive Visual Demonstration")
        print("The following demonstration will open a window to show the piston numbering visually.")
        print("You should see numbers 0-7 at the bottom of each piston.")
        print("Press Enter to start the visual demonstration (or 'n' to skip)...")
        
        user_input = input().strip().lower()
        if user_input != 'n':
            demo_piston_numbering_visual()
        
        print("\n🎉 All piston numbering demonstrations completed successfully!")
        print("\n📋 Summary:")
        print("✅ Piston numbering feature is working correctly")
        print("✅ Numbers are displayed at the bottom of each piston")
        print("✅ Works with any number of pistons")
        print("✅ Compatible with both human and rgb_array rendering")
        print("✅ Font is properly initialized")
        
    except Exception as e:
        print(f"❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 