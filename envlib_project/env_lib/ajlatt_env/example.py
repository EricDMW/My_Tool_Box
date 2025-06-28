#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of TTENV
"""

import numpy as np
import ttenv
import gym
import torch
import matplotlib.pyplot as plt
from ttenv.env import TargetTrackingEnv

def main():
    """Run a simple example of the target tracking environment."""
    
    # Register the environment
    ttenv.register()
    
    # Create environment with custom parameters
    env = ttenv.make(
        num_robots=3,
        num_targets=1,
        map_name='empty',
        sensor_range=3.0,
        communication_range=6.0,
        render=True,
        render_backend='TkAgg',
        seed=42
    )
    
    print(f"Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Number of robots: {env.num_robots}")
    print(f"Number of targets: {env.num_targets}")
    
    # Reset the environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Run a simple episode
    total_reward = 0
    for step in range(100):
        # Generate random actions for all robots
        action = env.action_space.sample()
        # Reshape to (num_robots, 2) if needed
        if len(action.shape) == 1:
            action = action.reshape(1, -1)
        
        # Take a step
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        total_reward += np.mean(reward)
        
        print(f"Step {step}: Reward = {np.mean(reward):.3f}, Done = {done}")
        
        # Render the environment
        env.render()
        
        # Check if episode is done
        if done:
            print(f"Episode finished after {step + 1} steps")
            break
    
    print(f"Total average reward: {total_reward:.3f}")
    
    # Close the environment
    env.close()

def render_target_trajectory():
    # Set up environment
    env = gym.make('TargetTracking-v0', map_name='obstacles05', render_mode='rgb_array')
    obs = env.reset()

    # For storing trajectory
    target_traj = []

    # Run the environment for a fixed number of steps
    for _ in range(300):
        # The environment will use the new target trajectory automatically
        result = env.step(np.zeros(env.action_space.shape))
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        # Get target position (assume info['target_state'] or similar)
        if 'target_state' in info:
            target_pos = info['target_state'][0][:2]
        else:
            # Fallback: try to extract from obs
            target_pos = obs['target_state'][0][:2] if 'target_state' in obs else None
        if target_pos is not None:
            target_traj.append(target_pos)
        if done:
            break

    # Plot the trajectory
    target_traj = np.array(target_traj)
    plt.figure(figsize=(8,8))
    img = env.render(mode='rgb_array')
    print(f"env.render(mode='rgb_array') returned type: {type(img)}, dtype: {getattr(img, 'dtype', None)}")
    if isinstance(img, np.ndarray) and img.dtype in [np.uint8, np.float32, np.float64, float]:
        plt.imshow(img)
    else:
        print("Warning: env.render(mode='rgb_array') did not return a valid image array. Skipping background image.")
    plt.plot(target_traj[:,0], target_traj[:,1], 'r-', label='Target Trajectory')
    plt.scatter(target_traj[0,0], target_traj[0,1], c='g', label='Start')
    plt.scatter(target_traj[-1,0], target_traj[-1,1], c='b', label='End')
    plt.legend()
    plt.title('Target Trajectory on obstacles05')
    plt.savefig('obstacles05_target_trajectory.png')
    plt.show()

if __name__ == "__main__":
    main()
    render_target_trajectory() 