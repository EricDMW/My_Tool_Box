#!/usr/bin/env python3
"""
Test script to verify TkAgg backend is working correctly.
Run this script to check if matplotlib can display interactive plots.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def test_tkagg_backend():
    """Test if TkAgg backend is working correctly."""
    
    print("Testing matplotlib backend configuration...")
    
    # Check current backend
    current_backend = matplotlib.get_backend()
    print(f"Current backend: {current_backend}")
    
    # Try to set TkAgg backend
    try:
        matplotlib.use('TkAgg', force=True)
        print("Successfully set TkAgg backend")
    except Exception as e:
        print(f"Failed to set TkAgg backend: {e}")
        return False
    
    # Create a simple test plot
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Generate some test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        # Plot data
        ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
        ax.plot(x, np.cos(x), 'r--', linewidth=2, label='cos(x)')
        
        # Add labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Test Plot - TkAgg Backend')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        print("✓ TkAgg backend is working correctly!")
        print("You should see an interactive plot window.")
        
        return True
        
    except Exception as e:
        print(f"Error creating test plot: {e}")
        return False

def test_environment_rendering():
    """Test if the environment can render with TkAgg."""
    
    print("\nTesting environment rendering...")
    
    try:
        # Import the environment
        from envlib_project.env_lib.ajlatt_env import ajlatt_env
        
        # Create environment with rendering enabled
        env = ajlatt_env(render=True, figID=0)
        
        # Reset environment
        obs = env.reset()
        
        # Take a few steps and render
        for i in range(5):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            
            # Render
            env.render()
            
            if done:
                break
        
        print("✓ Environment rendering is working!")
        return True
        
    except Exception as e:
        print(f"Error testing environment rendering: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TkAgg Backend Test")
    print("=" * 50)
    
    # Test basic TkAgg functionality
    tkagg_working = test_tkagg_backend()
    
    if tkagg_working:
        # Test environment rendering
        env_working = test_environment_rendering()
        
        if env_working:
            print("\n" + "=" * 50)
            print("✓ All tests passed! TkAgg backend is working correctly.")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("✗ Environment rendering test failed.")
            print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("✗ TkAgg backend test failed.")
        print("=" * 50) 