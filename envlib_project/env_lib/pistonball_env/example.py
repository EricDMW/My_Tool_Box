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
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action values: 0=down, 1=stay, 2=up for each piston")
    
    # Test specific discrete actions
    test_actions = [
        np.array([0, 0, 0]),  # All pistons down
        np.array([1, 1, 1]),  # All pistons stay
        np.array([2, 2, 2]),  # All pistons up
        np.array([0, 1, 2]),  # down, stay, up
        np.array([2, 1, 0]),  # up, stay, down
    ]
    
    # Run for each test action
    for i, action in enumerate(test_actions):
        print(f"Step {i + 1}: Action = {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward = {reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # Test random discrete actions
    print("\nTesting random discrete actions:")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Random action {step + 1}: {action}, Reward = {reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Discrete actions example completed!\n")


def movement_penalty_example():
    """Example demonstrating movement penalty functionality."""
    print("=== Movement Penalty Example ===")
    
    # Create two environments: one with penalty, one without
    env_with_penalty = PistonballEnv(
        n_pistons=5,
        movement_penalty=-0.1,
        movement_penalty_threshold=0.01,
        render_mode="rgb_array",
        continuous=True
    )
    
    env_no_penalty = PistonballEnv(
        n_pistons=5,
        movement_penalty=0.0,
        render_mode="rgb_array",
        continuous=True
    )
    
    print(f"Environment with penalty: {env_with_penalty.movement_penalty}")
    print(f"Environment without penalty: {env_no_penalty.movement_penalty}")
    
    # Reset both environments
    obs1, info1 = env_with_penalty.reset()
    obs2, info2 = env_no_penalty.reset()
    
    # Run both environments with same actions
    total_reward_with_penalty = 0
    total_reward_no_penalty = 0
    
    print("\nRunning both environments with same actions...")
    for step in range(30):
        # Generate same action for both environments
        action = np.random.uniform(-1, 1, 5)
        
        # Step with penalty
        obs1, reward1, terminated1, truncated1, info1 = env_with_penalty.step(action)
        total_reward_with_penalty += reward1
        
        # Step without penalty
        obs2, reward2, terminated2, truncated2, info2 = env_no_penalty.step(action)
        total_reward_no_penalty += reward2
        
        if step % 10 == 0:
            print(f"Step {step}: With penalty = {reward1:.3f}, Without penalty = {reward2:.3f}")
        
        # Reset if needed
        if terminated1 or truncated1:
            obs1, info1 = env_with_penalty.reset()
        if terminated2 or truncated2:
            obs2, info2 = env_no_penalty.reset()
    
    print(f"\nTotal rewards:")
    print(f"With penalty: {total_reward_with_penalty:.3f}")
    print(f"Without penalty: {total_reward_no_penalty:.3f}")
    print(f"Difference: {total_reward_with_penalty - total_reward_no_penalty:.3f}")
    
    env_with_penalty.close()
    env_no_penalty.close()
    print("Movement penalty example completed!\n")


def termination_condition_example():
    """Example demonstrating termination condition functionality."""
    print("=== Termination Condition Example ===")
    
    # Create two environments: one with termination, one without
    env_with_termination = PistonballEnv(
        n_pistons=5,
        terminated_condition=True,
        render_mode="rgb_array",
        continuous=True
    )
    
    env_no_termination = PistonballEnv(
        n_pistons=5,
        terminated_condition=False,
        render_mode="rgb_array",
        continuous=True
    )
    
    print(f"Environment with termination: {env_with_termination.terminated_condition}")
    print(f"Environment without termination: {env_no_termination.terminated_condition}")
    
    # Reset both environments
    obs1, info1 = env_with_termination.reset()
    obs2, info2 = env_no_termination.reset()
    
    # Run both environments with actions that move ball left
    print("\nRunning both environments with actions that move ball left...")
    for step in range(50):
        # Move all pistons up to create a path for the ball
        action = np.ones(5)  # All pistons up
        
        # Step with termination
        obs1, reward1, terminated1, truncated1, info1 = env_with_termination.step(action)
        
        # Step without termination
        obs2, reward2, terminated2, truncated2, info2 = env_no_termination.step(action)
        
        if step % 10 == 0:
            print(f"Step {step}: With termination = {reward1:.3f}, Without termination = {reward2:.3f}")
        
        # Check termination status
        if terminated1:
            print(f"✅ Environment with termination: Episode ended at step {step} (ball hit left wall)")
            print(f"   Final reward: {reward1:.3f}")
            obs1, info1 = env_with_termination.reset()
        
        if terminated2:
            print(f"❌ Environment without termination: Unexpected termination at step {step}")
            obs2, info2 = env_no_termination.reset()
        
        if truncated1:
            print(f"⏰ Environment with termination: Episode truncated at step {step}")
            obs1, info1 = env_with_termination.reset()
        
        if truncated2:
            print(f"⏰ Environment without termination: Episode truncated at step {step}")
            obs2, info2 = env_no_termination.reset()
    
    # Test different kappa values for termination rewards
    print("\n=== Testing Different Kappa Values ===")
    
    for kappa in [1, 2]:
        print(f"\nTesting kappa={kappa}")
        env_kappa = PistonballEnv(
            n_pistons=8,
            terminated_condition=True,
            kappa=kappa,
            render_mode="rgb_array",
            continuous=True
        )
        
        obs, info = env_kappa.reset()
        
        for step in range(50):
            action = np.ones(8)  # All pistons up
            obs, reward, terminated, truncated, info = env_kappa.step(action)
            
            if terminated:
                print(f"   ✅ Episode terminated at step {step}")
                print(f"   ✅ Final reward: {reward:.3f}")
                print(f"   ✅ Kappa={kappa} affects which pistons get termination reward")
                break
            elif truncated:
                print(f"   ⏰ Episode truncated at step {step}")
                break
        
        env_kappa.close()
    
    env_with_termination.close()
    env_no_termination.close()
    print("Termination condition example completed!\n")


def leftmost_piston_reward_example():
    """Example demonstrating leftmost piston reward functionality."""
    print("=== Leftmost Piston Reward Example ===")
    
    # Create environments with different leftmost piston rewards
    env_no_reward = PistonballEnv(
        n_pistons=5,
        terminated_condition=True,
        leftmost_piston_reward=0.0,  # No leftmost piston reward
        render_mode="rgb_array",
        continuous=True
    )
    
    env_positive_reward = PistonballEnv(
        n_pistons=5,
        terminated_condition=True,
        leftmost_piston_reward=1.0,  # Positive leftmost piston reward
        render_mode="rgb_array",
        continuous=True
    )
    
    env_negative_reward = PistonballEnv(
        n_pistons=5,
        terminated_condition=True,
        leftmost_piston_reward=-0.5,  # Negative leftmost piston reward
        render_mode="rgb_array",
        continuous=True
    )
    
    print(f"Environment with no leftmost reward: {env_no_reward.leftmost_piston_reward}")
    print(f"Environment with positive leftmost reward: {env_positive_reward.leftmost_piston_reward}")
    print(f"Environment with negative leftmost reward: {env_negative_reward.leftmost_piston_reward}")
    
    # Reset all environments
    obs1, info1 = env_no_reward.reset()
    obs2, info2 = env_positive_reward.reset()
    obs3, info3 = env_negative_reward.reset()
    
    # Run all environments with actions that move ball left
    print("\nRunning all environments with actions that move ball left...")
    for step in range(50):
        # Move all pistons up to create a path for the ball
        action = np.ones(5)  # All pistons up
        
        # Step all environments
        obs1, reward1, terminated1, truncated1, info1 = env_no_reward.step(action)
        obs2, reward2, terminated2, truncated2, info2 = env_positive_reward.step(action)
        obs3, reward3, terminated3, truncated3, info3 = env_negative_reward.step(action)
        
        if step % 10 == 0:
            print(f"Step {step}: No reward = {reward1:.3f}, Positive = {reward2:.3f}, Negative = {reward3:.3f}")
        
        # Check termination status for all environments
        if terminated1:
            print(f"✅ Environment with no leftmost reward: Episode ended at step {step}")
            print(f"   Final reward: {reward1:.3f}")
            print(f"   Local rewards: {info1['local_rewards']}")
            obs1, info1 = env_no_reward.reset()
        
        if terminated2:
            print(f"✅ Environment with positive leftmost reward: Episode ended at step {step}")
            print(f"   Final reward: {reward2:.3f}")
            print(f"   Local rewards: {info2['local_rewards']}")
            print(f"   Leftmost piston reward: {info2['local_rewards'][0]}")
            print(f"   Other piston rewards: {info2['local_rewards'][1:]}")
            obs2, info2 = env_positive_reward.reset()
        
        if terminated3:
            print(f"✅ Environment with negative leftmost reward: Episode ended at step {step}")
            print(f"   Final reward: {reward3:.3f}")
            print(f"   Local rewards: {info3['local_rewards']}")
            print(f"   Leftmost piston reward: {info3['local_rewards'][0]}")
            print(f"   Other piston rewards: {info3['local_rewards'][1:]}")
            obs3, info3 = env_negative_reward.reset()
        
        if truncated1 or truncated2 or truncated3:
            print(f"⏰ Episode truncated at step {step}")
            if truncated1:
                obs1, info1 = env_no_reward.reset()
            if truncated2:
                obs2, info2 = env_positive_reward.reset()
            if truncated3:
                obs3, info3 = env_negative_reward.reset()
    
    # Test different leftmost piston reward values
    print("\n=== Testing Different Leftmost Piston Reward Values ===")
    
    reward_values = [0.5, 1.0, 2.0]
    for reward_val in reward_values:
        print(f"\nTesting leftmost_piston_reward={reward_val}")
        env_test = PistonballEnv(
            n_pistons=6,
            terminated_condition=True,
            leftmost_piston_reward=reward_val,
            render_mode="rgb_array",
            continuous=True
        )
        
        obs, info = env_test.reset()
        
        for step in range(50):
            action = np.ones(6)  # All pistons up
            obs, reward, terminated, truncated, info = env_test.step(action)
            
            if terminated:
                print(f"   ✅ Episode terminated at step {step}")
                print(f"   ✅ Final reward: {reward:.3f}")
                print(f"   ✅ Leftmost piston reward: {info['local_rewards'][0]:.3f}")
                print(f"   ✅ Expected leftmost piston reward: {0.5 + reward_val:.3f}")
                break
            elif truncated:
                print(f"   ⏰ Episode truncated at step {step}")
                break
        
        env_test.close()
    
    env_no_reward.close()
    env_positive_reward.close()
    env_negative_reward.close()
    print("Leftmost piston reward example completed!\n")


def termination_reward_example():
    """Example demonstrating configurable termination reward functionality."""
    print("=== Configurable Termination Reward Example ===")
    
    # Create environments with different termination rewards
    env_default = PistonballEnv(
        n_pistons=5,
        terminated_condition=True,
        render_mode="rgb_array",
        continuous=True
    )
    
    env_high_reward = PistonballEnv(
        n_pistons=5,
        terminated_condition=True,
        termination_reward=1.0,  # High termination reward
        render_mode="rgb_array",
        continuous=True
    )
    
    env_low_reward = PistonballEnv(
        n_pistons=5,
        terminated_condition=True,
        termination_reward=0.1,  # Low termination reward
        render_mode="rgb_array",
        continuous=True
    )
    
    print(f"Environment with default termination reward: {env_default.termination_reward}")
    print(f"Environment with high termination reward: {env_high_reward.termination_reward}")
    print(f"Environment with low termination reward: {env_low_reward.termination_reward}")
    
    # Reset all environments
    obs1, info1 = env_default.reset()
    obs2, info2 = env_high_reward.reset()
    obs3, info3 = env_low_reward.reset()
    
    # Run all environments with actions that move ball left
    print("\nRunning all environments with actions that move ball left...")
    for step in range(50):
        # Move all pistons up to create a path for the ball
        action = np.ones(5)  # All pistons up
        
        # Step all environments
        obs1, reward1, terminated1, truncated1, info1 = env_default.step(action)
        obs2, reward2, terminated2, truncated2, info2 = env_high_reward.step(action)
        obs3, reward3, terminated3, truncated3, info3 = env_low_reward.step(action)
        
        if step % 10 == 0:
            print(f"Step {step}: Default = {reward1:.3f}, High = {reward2:.3f}, Low = {reward3:.3f}")
        
        # Check termination status for all environments
        if terminated1:
            print(f"✅ Environment with default termination reward: Episode ended at step {step}")
            print(f"   Final reward: {reward1:.3f}")
            print(f"   Local rewards: {info1['local_rewards']}")
            obs1, info1 = env_default.reset()
        
        if terminated2:
            print(f"✅ Environment with high termination reward: Episode ended at step {step}")
            print(f"   Final reward: {reward2:.3f}")
            print(f"   Local rewards: {info2['local_rewards']}")
            obs2, info2 = env_high_reward.reset()
        
        if terminated3:
            print(f"✅ Environment with low termination reward: Episode ended at step {step}")
            print(f"   Final reward: {reward3:.3f}")
            print(f"   Local rewards: {info3['local_rewards']}")
            obs3, info3 = env_low_reward.reset()
        
        if truncated1 or truncated2 or truncated3:
            print(f"⏰ Episode truncated at step {step}")
            if truncated1:
                obs1, info1 = env_default.reset()
            if truncated2:
                obs2, info2 = env_high_reward.reset()
            if truncated3:
                obs3, info3 = env_low_reward.reset()
    
    # Test different termination reward values
    print("\n=== Testing Different Termination Reward Values ===")
    
    termination_rewards = [0.1, 0.5, 1.0, 2.0]
    for term_reward in termination_rewards:
        print(f"\nTesting termination_reward={term_reward}")
        env_test = PistonballEnv(
            n_pistons=6,
            terminated_condition=True,
            termination_reward=term_reward,
            render_mode="rgb_array",
            continuous=True
        )
        
        obs, info = env_test.reset()
        
        for step in range(50):
            action = np.ones(6)  # All pistons up
            obs, reward, terminated, truncated, info = env_test.step(action)
            
            if terminated:
                print(f"   ✅ Episode terminated at step {step}")
                print(f"   ✅ Final reward: {reward:.3f}")
                print(f"   ✅ Termination reward: {term_reward}")
                break
            elif truncated:
                print(f"   ⏰ Episode truncated at step {step}")
                break
        
        env_test.close()
    
    env_default.close()
    env_high_reward.close()
    env_low_reward.close()
    print("Configurable termination reward example completed!\n")


def multi_piston_control_example():
    """Example demonstrating multi-piston control patterns."""
    print("=== Multi-Piston Control Example ===")
    
    env = PistonballEnv(
        n_pistons=6,
        render_mode="rgb_array",
        continuous=True
    )
    
    obs, info = env.reset()
    print(f"Environment with {env.n_pistons} pistons")
    print(f"Action space: {env.action_space}")
    
    # Define different control patterns
    patterns = [
        ("All pistons up", np.ones(6)),
        ("All pistons down", -np.ones(6)),
        ("Stay still", np.zeros(6)),
        ("Alternating", np.array([1, -1, 1, -1, 1, -1])),
        ("Wave pattern", np.array([1, 0.5, 0, -0.5, -1, -0.5])),
        ("Random pattern", np.random.uniform(-1, 1, 6))
    ]
    
    print("\nTesting different control patterns:")
    for pattern_name, action in patterns:
        print(f"\nPattern: {pattern_name}")
        print(f"Action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Multi-piston control example completed!\n")


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
        movement_penalty_example()
        termination_condition_example()
        leftmost_piston_reward_example()
        termination_reward_example()
        multi_piston_control_example()
        manual_control_example()
        
        print("All examples completed successfully!")
        print("\nTo test the environment more thoroughly, run:")
        print("python test_env.py")
        print("python test_movement_penalty.py")
        print("python test_termination_condition.py")
        print("python test_leftmost_piston_reward.py")
        print("python test_termination_reward.py")
        
    except Exception as e:
        print(f"Error during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 