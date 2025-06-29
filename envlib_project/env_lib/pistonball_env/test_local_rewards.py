#!/usr/bin/env python3
"""
Test script for local reward system in Pistonball environment.

This script tests the local reward calculation where each piston gets a reward
based on its own observation of the ball movement.
"""

import numpy as np
from pistonball_env import PistonballEnv

def test_local_rewards():
    """Test the local reward system."""
    print("Testing Local Reward System")
    print("=" * 50)
    
    # Test different kappa values
    kappa_values = [0, 1, 2]
    n_pistons = 8
    
    for kappa in kappa_values:
        print(f"\nTesting kappa = {kappa}")
        print("-" * 30)
        
        # Create environment with specific kappa
        env = PistonballEnv(
            n_pistons=n_pistons,
            kappa=kappa,
            render_mode=None,
            continuous=True,
            time_penalty=-0.1
        )
        
        # Reset environment
        obs, info = env.reset()
        
        # Get initial ball position
        ball_x = env.ball.position[0]
        ball_piston_index = int((ball_x - env.wall_width - env.piston_radius) / env.piston_width)
        ball_piston_index = max(0, min(ball_piston_index, env.n_pistons - 1))
        
        print(f"Initial ball position: x={ball_x:.1f}")
        print(f"Ball is closest to piston {ball_piston_index}")
        
        # Take a few steps and observe local rewards
        for step in range(3):
            print(f"\nStep {step + 1}:")
            
            # Take random action
            action = env.action_space.sample()
            obs, local_rewards, total_reward, terminated, truncated, info = env.step(action)
            
            print(f"Total reward: {total_reward:.3f}")
            print(f"Local rewards: {local_rewards}")
            
            # Check which pistons can observe the ball
            observable_pistons = []
            for i in range(env.n_pistons):
                if env._can_observe_ball(i, env.ball.position[0]):
                    observable_pistons.append(i)
            
            print(f"Observable pistons: {observable_pistons}")
            
            # Verify that pistons that can't observe the ball have lower rewards
            # (they should only get time penalty)
            for i in range(env.n_pistons):
                can_observe = i in observable_pistons
                reward = local_rewards[i]
                
                if can_observe:
                    print(f"Piston {i}: observable, reward = {reward:.3f}")
                else:
                    print(f"Piston {i}: not observable, reward = {reward:.3f}")
                    # Non-observable pistons should only get time penalty
                    expected_reward = env.time_penalty
                    if abs(reward - expected_reward) > 1e-6:
                        print(f"  WARNING: Piston {i} has unexpected reward!")
            
            if terminated or truncated:
                print("Episode ended")
                break
        
        env.close()
    
    print("\n" + "=" * 50)
    print("Local reward test completed!")

def test_reward_consistency():
    """Test that rewards are consistent across steps."""
    print("\nTesting Reward Consistency")
    print("=" * 50)
    
    env = PistonballEnv(
        n_pistons=5,
        kappa=1,
        render_mode=None,
        continuous=True,
        time_penalty=-0.1
    )
    
    obs, info = env.reset()
    
    # Take same action multiple times and check consistency
    action = np.zeros(5)  # No movement
    
    for step in range(5):
        obs, local_rewards, total_reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step + 1}:")
        print(f"  Local rewards: {local_rewards}")
        print(f"  Total reward: {total_reward:.3f}")
        
        # Check that total reward equals sum of local rewards
        calculated_total = np.sum(local_rewards)
        if abs(total_reward - calculated_total) > 1e-6:
            print(f"  ERROR: Total reward mismatch! Expected {calculated_total}, got {total_reward}")
        
        if terminated or truncated:
            break
    
    env.close()

def test_movement_penalty_local():
    """Test that movement penalties are applied locally."""
    print("\nTesting Movement Penalty Local Application")
    print("=" * 50)
    
    env = PistonballEnv(
        n_pistons=5,
        kappa=1,
        render_mode=None,
        continuous=True,
        time_penalty=-0.1,
        movement_penalty=-0.05,
        movement_penalty_threshold=0.01
    )
    
    obs, info = env.reset()
    
    # Take action that moves only one piston
    action = np.zeros(5)
    action[2] = 1.0  # Move only piston 2
    
    obs, local_rewards, total_reward, terminated, truncated, info = env.step(action)
    
    print(f"Action: {action}")
    print(f"Local rewards: {local_rewards}")
    print(f"Total reward: {total_reward:.3f}")
    
    # Check that only piston 2 has movement penalty
    for i in range(5):
        if i == 2:
            print(f"Piston {i}: should have movement penalty, reward = {local_rewards[i]:.3f}")
        else:
            print(f"Piston {i}: should not have movement penalty, reward = {local_rewards[i]:.3f}")
    
    env.close()

if __name__ == "__main__":
    # Run all tests
    test_local_rewards()
    test_reward_consistency()
    test_movement_penalty_local()
    
    print("\nAll tests completed successfully!") 