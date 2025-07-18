#!/usr/bin/env python3
"""
Example showing real-time rendering during training progress.
"""

import numpy as np
from wireless_comm_env import WirelessCommEnv
import time

def train_with_realtime_rendering():
    """Train the environment with real-time rendering."""
    print("Starting training with real-time rendering...")
    
    # Create environment
    env = WirelessCommEnv(grid_x=4, grid_y=4, render_mode="rgb_array")
    obs, info = env.reset()
    
    total_reward = 0
    render_frequency = 5  # Render every 5 steps
    
    for step in range(50):  # 50 training steps
        # Take action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
        
        # Render in real-time every few steps
        if (step + 1) % render_frequency == 0:
            print(f"Rendering step {step + 1}...")
            env.render_realtime(show=True)
            time.sleep(0.5)  # Pause to see the rendering
        
        # Reset if episode is done
        if terminated or truncated:
            print(f"Episode finished at step {step + 1}")
            obs, info = env.reset()
            total_reward = 0
    
    env.close()
    print("Training completed!")

def train_with_save_frames():
    """Train and save specific frames for analysis."""
    print("Starting training with frame saving...")
    
    env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="rgb_array")
    obs, info = env.reset()
    
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Save specific frames
        if step in [0, 5, 10, 15, 19]:  # Save frames at specific steps
            save_path = f"frame_step_{step + 1}.png"
            env.render_realtime(show=False, save_path=save_path)
            print(f"Saved frame at step {step + 1} to {save_path}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Training with frame saving completed!")

if __name__ == "__main__":
    print("Choose training mode:")
    print("1. Real-time rendering during training")
    print("2. Save specific frames during training")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        train_with_realtime_rendering()
    else:
        train_with_save_frames() 