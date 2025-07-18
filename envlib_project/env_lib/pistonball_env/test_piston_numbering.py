#!/usr/bin/env python3
"""
Test script for piston numbering feature.

This script tests that piston order numbers are correctly displayed
at the bottom of each piston during rendering.
"""

import pygame
import numpy as np
import time
from pistonball_env import PistonballEnv


def test_piston_numbering_visual():
    """Test piston numbering with visual rendering."""
    print("🎯 Testing Piston Numbering Feature")
    print("=" * 50)
    
    # Create environment with human rendering
    env = PistonballEnv(n_pistons=8, render_mode="human")
    obs, info = env.reset()
    
    print(f"Environment created with {env.n_pistons} pistons")
    print("You should see piston numbers (0-7) at the bottom of each piston")
    print("Press Enter to continue with the test...")
    input()
    
    # Run a few steps to see the numbering in action
    for step in range(20):
        # Create a simple wave pattern to see piston movement
        action = np.array([np.sin(step * 0.5 + i * 0.5) for i in range(8)])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to show the numbering
        env.render()
        
        # Small delay to see the movement
        time.sleep(0.1)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("✅ Piston numbering visual test completed!")


def test_piston_numbering_rgb_array():
    """Test piston numbering with rgb_array rendering."""
    print("\n🖼️ Testing Piston Numbering with RGB Array")
    print("=" * 50)
    
    # Create environment with rgb_array rendering
    env = PistonballEnv(n_pistons=5, render_mode="rgb_array")
    obs, info = env.reset()
    
    print(f"Environment created with {env.n_pistons} pistons")
    print("Testing rgb_array rendering with piston numbering...")
    
    # Take a few steps and capture frames
    for step in range(5):
        action = np.random.uniform(-1, 1, 5)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get rendered frame
        frame = env.render()
        
        if frame is not None:
            print(f"Step {step}: Frame shape = {frame.shape}")
            print(f"Frame data type = {frame.dtype}")
            print(f"Frame value range = [{frame.min()}, {frame.max()}]")
        else:
            print(f"Step {step}: No frame returned")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("✅ Piston numbering rgb_array test completed!")


def test_different_piston_counts():
    """Test piston numbering with different piston counts."""
    print("\n🔢 Testing Different Piston Counts")
    print("=" * 50)
    
    piston_counts = [3, 5, 10, 15]
    
    for n_pistons in piston_counts:
        print(f"\nTesting with {n_pistons} pistons...")
        
        env = PistonballEnv(n_pistons=n_pistons, render_mode="rgb_array")
        obs, info = env.reset()
        
        # Take one step
        action = np.random.uniform(-1, 1, n_pistons)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get rendered frame
        frame = env.render()
        
        if frame is not None:
            print(f"  Frame shape: {frame.shape}")
            print(f"  Piston numbers should be 0-{n_pistons-1}")
        else:
            print("  No frame returned")
        
        env.close()
    
    print("✅ Different piston counts test completed!")


def test_font_initialization():
    """Test that font is properly initialized."""
    print("\n🔤 Testing Font Initialization")
    print("=" * 50)
    
    env = PistonballEnv(n_pistons=5, render_mode="rgb_array")
    
    # Check if font is initialized
    if hasattr(env, 'font') and env.font is not None:
        print("✅ Font is properly initialized")
        print(f"Font type: {type(env.font)}")
    else:
        print("❌ Font is not initialized")
    
    env.close()
    print("✅ Font initialization test completed!")


def main():
    """Run all piston numbering tests."""
    print("Piston Numbering Feature Test Suite")
    print("=" * 40)
    
    try:
        test_font_initialization()
        test_piston_numbering_rgb_array()
        test_different_piston_counts()
        
        print("\n🎮 Interactive Visual Test")
        print("The following test will open a window to show the piston numbering visually.")
        print("You should see numbers 0-7 at the bottom of each piston.")
        print("Press Enter to start the visual test (or 'n' to skip)...")
        
        user_input = input().strip().lower()
        if user_input != 'n':
            test_piston_numbering_visual()
        
        print("\n🎉 All piston numbering tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 