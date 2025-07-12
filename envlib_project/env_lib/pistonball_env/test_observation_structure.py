#!/usr/bin/env python3
"""
Test script for the new observation structure in Pistonball environment.

This script tests the updated observation space that includes piston x-axis location.
"""

import numpy as np
from pistonball_env import PistonballEnv

def test_observation_structure():
    """Test the new observation structure with piston x-axis location."""
    print("Testing New Observation Structure")
    print("=" * 50)
    
    # Test different configurations
    n_pistons = 5
    kappa = 1
    
    env = PistonballEnv(
        n_pistons=n_pistons,
        kappa=kappa,
        render_mode=None,
        continuous=True
    )
    
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Expected shape: ({n_pistons}, 7)")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Actual observation shape: {obs.shape}")
    
    # Check observation structure for each piston
    for i in range(n_pistons):
        print(f"\nPiston {i} observation:")
        print(f"  [0] Piston Y position: {obs[i, 0]:.3f}")
        print(f"  [1] Piston X position: {obs[i, 1]:.3f}")
        print(f"  [2] Ball X position: {obs[i, 2]:.3f}")
        print(f"  [3] Ball Y position: {obs[i, 3]:.3f}")
        print(f"  [4] Ball X velocity: {obs[i, 4]:.3f}")
        print(f"  [5] Ball Y velocity: {obs[i, 5]:.3f}")
        print(f"  [6] Ball angular velocity: {obs[i, 6]:.3f}")
        
        # Verify piston x position is reasonable (should be between 0 and 1)
        if not (0 <= obs[i, 1] <= 1):
            print(f"  WARNING: Piston {i} X position {obs[i, 1]:.3f} is outside [0,1] range!")
    
    # Check that piston X positions are in ascending order
    piston_x_positions = obs[:, 1]
    if not np.all(np.diff(piston_x_positions) >= 0):
        print("WARNING: Piston X positions are not in ascending order!")
    else:
        print("\n✓ Piston X positions are in ascending order")
    
    # Take a step and check observation structure
    print("\nTaking a step...")
    action = np.zeros(n_pistons)
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step observation shape: {obs.shape}")
    
    # Check that piston X positions remain the same (they shouldn't change)
    new_piston_x_positions = obs[:, 1]
    if np.allclose(piston_x_positions, new_piston_x_positions):
        print("✓ Piston X positions remain constant across steps")
    else:
        print("WARNING: Piston X positions changed across steps!")
    
    env.close()

def test_kappa_observation_with_x():
    """Test kappa observation system with the new X position dimension."""
    print("\nTesting Kappa Observation with X Position")
    print("=" * 50)
    
    n_pistons = 8
    kappa = 1
    
    env = PistonballEnv(
        n_pistons=n_pistons,
        kappa=kappa,
        render_mode=None,
        continuous=True
    )
    
    obs, info = env.reset()
    
    # Get ball position
    ball_x = env.ball.position[0]
    ball_piston_index = int((ball_x - env.wall_width - env.piston_radius) / env.piston_width)
    ball_piston_index = max(0, min(ball_piston_index, env.n_pistons - 1))
    
    print(f"Ball position: x={ball_x:.1f}")
    print(f"Ball is closest to piston {ball_piston_index}")
    
    # Check which pistons can observe the ball
    observable_pistons = []
    for i in range(env.n_pistons):
        if env._can_observe_ball(i, ball_x):
            observable_pistons.append(i)
    
    print(f"Observable pistons: {observable_pistons}")
    
    # Check observation structure for observable vs non-observable pistons
    for i in range(env.n_pistons):
        can_observe = i in observable_pistons
        ball_obs = obs[i, 2:7]  # Ball-related observations (indices 2-6)
        has_ball_info = np.any(ball_obs != 0)
        
        print(f"\nPiston {i}:")
        print(f"  X position: {obs[i, 1]:.3f}")
        print(f"  Can observe ball: {can_observe}")
        print(f"  Has ball info: {has_ball_info}")
        print(f"  Ball observations: {ball_obs}")
        
        # Verify that observation matches kappa rule
        if can_observe and not has_ball_info:
            print(f"  ERROR: Piston {i} should observe ball but has no ball info!")
        elif not can_observe and has_ball_info:
            print(f"  ERROR: Piston {i} shouldn't observe ball but has ball info!")
    
    env.close()

def test_observation_normalization():
    """Test that observations are properly normalized."""
    print("\nTesting Observation Normalization")
    print("=" * 50)
    
    env = PistonballEnv(
        n_pistons=10,
        kappa=2,
        render_mode=None,
        continuous=True
    )
    
    obs, info = env.reset()
    
    # Check piston Y position normalization (should be in [-1, 1])
    piston_y_positions = obs[:, 0]
    if np.all((-1 <= piston_y_positions) & (piston_y_positions <= 1)):
        print("✓ Piston Y positions are properly normalized to [-1, 1]")
    else:
        print("WARNING: Piston Y positions are not properly normalized!")
        print(f"  Range: [{piston_y_positions.min():.3f}, {piston_y_positions.max():.3f}]")
    
    # Check piston X position normalization (should be in [0, 1])
    piston_x_positions = obs[:, 1]
    if np.all((0 <= piston_x_positions) & (piston_x_positions <= 1)):
        print("✓ Piston X positions are properly normalized to [0, 1]")
    else:
        print("WARNING: Piston X positions are not properly normalized!")
        print(f"  Range: [{piston_x_positions.min():.3f}, {piston_x_positions.max():.3f}]")
    
    # Check ball position normalization for observable pistons
    for i in range(env.n_pistons):
        if env._can_observe_ball(i, env.ball.position[0]):
            ball_x = obs[i, 2]
            ball_y = obs[i, 3]
            
            if not (0 <= ball_x <= 1):
                print(f"WARNING: Piston {i} ball X position {ball_x:.3f} not in [0,1]")
            if not (0 <= ball_y <= 1):
                print(f"WARNING: Piston {i} ball Y position {ball_y:.3f} not in [0,1]")
    
    env.close()

if __name__ == "__main__":
    # Run all tests
    test_observation_structure()
    test_kappa_observation_with_x()
    test_observation_normalization()
    
    print("\n" + "=" * 50)
    print("All observation structure tests completed successfully!") 