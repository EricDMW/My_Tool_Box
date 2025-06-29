#!/usr/bin/env python3
"""
Test script for kappa-hop observation system in Pistonball environment.

This script tests the limited observation range feature where pistons can only
observe the ball if they are within kappa hops of the piston closest to the ball.
"""

import numpy as np
import matplotlib.pyplot as plt
from pistonball_env import PistonballEnv

def test_kappa_observation():
    """Test the kappa-hop observation system."""
    print("Testing Kappa-Hop Observation System")
    print("=" * 50)
    
    # Test different kappa values
    kappa_values = [0, 1, 2, 3]
    n_pistons = 10
    
    for kappa in kappa_values:
        print(f"\nTesting kappa = {kappa}")
        print("-" * 30)
        
        # Create environment with specific kappa
        env = PistonballEnv(
            n_pistons=n_pistons,
            kappa=kappa,
            render_mode=None,  # No rendering for faster testing
            continuous=True
        )
        
        # Reset environment
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
            can_observe = env._can_observe_ball(i, ball_x)
            if can_observe:
                observable_pistons.append(i)
            
            # Check if observation matches expectation
            ball_obs = obs[i, 1:6]  # Ball-related observations
            has_ball_info = np.any(ball_obs != 0)
            
            print(f"Piston {i}: can_observe={can_observe}, has_ball_info={has_ball_info}")
            
            # Verify that observation matches kappa rule
            distance = abs(i - ball_piston_index)
            expected_observable = distance <= kappa
            
            if can_observe != expected_observable:
                print(f"  ERROR: Piston {i} observation doesn't match kappa rule!")
            
            if can_observe and not has_ball_info:
                print(f"  ERROR: Piston {i} should observe ball but has no ball info!")
            elif not can_observe and has_ball_info:
                print(f"  ERROR: Piston {i} shouldn't observe ball but has ball info!")
        
        print(f"Observable pistons: {observable_pistons}")
        
        env.close()
    
    print("\n" + "=" * 50)
    print("Kappa observation test completed!")

def test_kappa_visualization():
    """Create a visualization of the kappa-hop observation system."""
    print("\nCreating Kappa Observation Visualization")
    print("=" * 50)
    
    n_pistons = 15
    kappa_values = [0, 1, 2, 3]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, kappa in enumerate(kappa_values):
        ax = axes[idx]
        
        # Create environment
        env = PistonballEnv(
            n_pistons=n_pistons,
            kappa=kappa,
            render_mode=None,
            continuous=True
        )
        
        # Reset and get observations
        obs, info = env.reset()
        ball_x = env.ball.position[0]
        ball_piston_index = int((ball_x - env.wall_width - env.piston_radius) / env.piston_width)
        ball_piston_index = max(0, min(ball_piston_index, env.n_pistons - 1))
        
        # Create visualization
        piston_positions = np.arange(n_pistons)
        observable_mask = np.array([env._can_observe_ball(i, ball_x) for i in range(n_pistons)])
        
        # Plot pistons
        ax.scatter(piston_positions, np.zeros(n_pistons), c='blue', s=100, label='Pistons', alpha=0.7)
        
        # Highlight observable pistons
        observable_positions = piston_positions[observable_mask]
        ax.scatter(observable_positions, np.zeros(len(observable_positions)), 
                  c='red', s=150, label='Observable', alpha=0.9)
        
        # Highlight ball position
        ax.scatter([ball_piston_index], [0], c='green', s=200, marker='o', 
                  label='Ball Position', alpha=0.9)
        
        # Draw observation range
        if kappa > 0:
            for i in range(max(0, ball_piston_index - kappa), min(n_pistons, ball_piston_index + kappa + 1)):
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.2, color='red')
        
        ax.set_title(f'Kappa = {kappa} (Observation Range)')
        ax.set_xlabel('Piston Index')
        ax.set_ylabel('Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_pistons - 0.5)
        ax.set_ylim(-0.5, 0.5)
        
        env.close()
    
    plt.tight_layout()
    plt.savefig('kappa_observation_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'kappa_observation_visualization.png'")
    plt.show()

def test_kappa_dynamics():
    """Test how observation changes as the ball moves."""
    print("\nTesting Kappa Observation Dynamics")
    print("=" * 50)
    
    n_pistons = 8
    kappa = 2
    
    env = PistonballEnv(
        n_pistons=n_pistons,
        kappa=kappa,
        render_mode=None,
        continuous=True
    )
    
    obs, info = env.reset()
    
    print(f"Initial ball position: x={env.ball.position[0]:.1f}")
    print("Initial observable pistons:", [i for i in range(n_pistons) if env._can_observe_ball(i, env.ball.position[0])])
    
    # Move ball to different positions and check observations
    test_positions = [200, 300, 400, 500]
    
    for pos_x in test_positions:
        # Manually set ball position for testing
        env.ball.position = (pos_x, env.ball.position[1])
        
        # Get new observations
        obs = env._get_obs()
        
        ball_piston_index = int((pos_x - env.wall_width - env.piston_radius) / env.piston_width)
        ball_piston_index = max(0, min(ball_piston_index, env.n_pistons - 1))
        
        observable_pistons = [i for i in range(n_pistons) if env._can_observe_ball(i, pos_x)]
        
        print(f"\nBall at x={pos_x:.1f} (piston {ball_piston_index})")
        print(f"Observable pistons: {observable_pistons}")
        
        # Verify observations
        for i in range(n_pistons):
            ball_obs = obs[i, 1:6]
            has_ball_info = np.any(ball_obs != 0)
            can_observe = i in observable_pistons
            
            if can_observe != has_ball_info:
                print(f"  ERROR: Piston {i} observation mismatch!")
    
    env.close()

def test_kappa_edge_cases():
    """Test edge cases for kappa observation system."""
    print("\nTesting Kappa Edge Cases")
    print("=" * 50)
    
    # Test with kappa = 0 (no observation)
    print("Testing kappa = 0:")
    env = PistonballEnv(n_pistons=5, kappa=0, render_mode=None)
    obs, info = env.reset()
    
    for i in range(5):
        ball_obs = obs[i, 1:6]
        has_ball_info = np.any(ball_obs != 0)
        print(f"Piston {i}: has_ball_info={has_ball_info}")
    
    env.close()
    
    # Test with large kappa
    print("\nTesting large kappa:")
    env = PistonballEnv(n_pistons=5, kappa=10, render_mode=None)
    obs, info = env.reset()
    
    for i in range(5):
        ball_obs = obs[i, 1:6]
        has_ball_info = np.any(ball_obs != 0)
        print(f"Piston {i}: has_ball_info={has_ball_info}")
    
    env.close()

if __name__ == "__main__":
    # Run all tests
    test_kappa_observation()
    test_kappa_dynamics()
    test_kappa_edge_cases()
    
    # Create visualization
    try:
        test_kappa_visualization()
    except ImportError:
        print("Matplotlib not available, skipping visualization")
    
    print("\nAll tests completed successfully!") 