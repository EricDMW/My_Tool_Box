"""
Simple example demonstrating the Pistonball environment.

This script shows how to create and use the pistonball environment
with different configurations.
"""

import numpy as np
from pistonball_env import PistonballEnv


def basic_example():
    """Basic example of using the pistonball environment."""
    print("=== Basic Pistonball Example ===")
    
    # Create environment with 5 pistons
    env = PistonballEnv(
        n_pistons=5,
        render_mode="rgb_array",
        continuous=True
    )
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Number of pistons: {env.n_pistons}")
    
    # Run for 50 steps
    total_reward = 0
    for step in range(50):
        # Take random actions
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
        
        # Reset if episode is done
        if terminated or truncated:
            print(f"Episode finished at step {step + 1}")
            obs, info = env.reset()
            total_reward = 0
    
    env.close()
    print("Basic example completed!\n")


def physics_example():
    """Example with different physics parameters."""
    print("=== Physics Parameters Example ===")
    
    # Create environment with heavy, bouncy ball
    env = PistonballEnv(
        n_pistons=8,
        ball_mass=2.0,        # Heavy ball
        ball_friction=0.1,    # Low friction (slippery)
        ball_elasticity=2.0,  # Very bouncy
        render_mode="rgb_array",
        continuous=True
    )
    
    obs, info = env.reset()
    print(f"Environment created with heavy, bouncy ball")
    print(f"Ball mass: {env.ball_mass}")
    print(f"Ball friction: {env.ball_friction}")
    print(f"Ball elasticity: {env.ball_elasticity}")
    
    # Run for 30 steps
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            if env.ball is not None:
                print(f"Step {step}: Ball position = ({env.ball.position[0]:.1f}, {env.ball.position[1]:.1f})")
            else:
                print(f"Step {step}: Ball not initialized")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Physics example completed!\n")


def discrete_actions_example():
    """Example with discrete actions."""
    print("=== Discrete Actions Example ===")
    
    # Create environment with discrete actions
    env = PistonballEnv(
        n_pistons=3,  # Fewer pistons for discrete actions
        continuous=False,
        render_mode="rgb_array"
    )
    
    obs, info = env.reset()
    print(f"Discrete action space: {env.action_space}")
    n_actions = getattr(env.action_space, 'n', None)
    if n_actions is not None:
        print(f"Number of possible actions: {n_actions}")
    else:
        print("Action space doesn't have 'n' attribute")
    
    # Run for 20 steps
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step + 1}: Action = {action}, Reward = {reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Discrete actions example completed!\n")


def manual_control_example():
    """Example showing how to set up manual control."""
    print("=== Manual Control Setup Example ===")
    print("This example shows how to set up manual control.")
    print("To actually control the environment, run the manual_policy.py file.")
    
    from manual_policy import ManualPolicy
    
    # Create environment
    env = PistonballEnv(
        n_pistons=6,
        render_mode="human",
        continuous=True
    )
    
    # Create manual policy
    manual_policy = ManualPolicy(env)
    
    print(f"Manual policy created for {env.n_pistons} pistons")
    print("Controls:")
    print("- W/S: Move selected piston up/down")
    print("- A/D: Select previous/next piston")
    print("- ESC: Exit")
    print("- BACKSPACE: Reset environment")
    
    env.close()
    print("Manual control setup example completed!\n")


def main():
    """Run all examples."""
    print("Pistonball Environment Examples")
    print("=" * 35)
    
    try:
        basic_example()
        physics_example()
        discrete_actions_example()
        manual_control_example()
        
        print("All examples completed successfully!")
        print("\nTo test the environment more thoroughly, run:")
        print("python test_env.py")
        
    except Exception as e:
        print(f"Error during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 