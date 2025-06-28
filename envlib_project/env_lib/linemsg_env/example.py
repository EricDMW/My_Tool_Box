"""
Example script demonstrating the Line Message environment.

This script shows how to create and use the line message environment
with different configurations and scenarios.
"""

import numpy as np
from linemsg_env import LineMsgEnv


def basic_example():
    """Basic example of using the line message environment."""
    print("=== Basic Line Message Example ===")
    
    # Create environment with 6 agents
    env = LineMsgEnv(
        num_agents=6,
        render_mode="human",
    )
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Number of agents: {env.num_agents}")
    
    # Run for 20 steps
    total_reward = 0
    for step in range(20):
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


def message_propagation_example():
    """Example demonstrating message propagation dynamics."""
    print("=== Message Propagation Example ===")
    
    # Create environment with more agents to see propagation
    env = LineMsgEnv(
        num_agents=10,
        render_mode="human",
    )
    
    obs, info = env.reset()
    print(f"Environment created with {env.num_agents} agents")
    if env.state is not None:
        print(f"Initial state: {env.state[env.n_obs_nghbr:-env.n_obs_nghbr]}")
    else:
        print("Initial state is not initialized.")
    
    # Track message propagation over time
    message_history = []
    
    # Run for 30 steps
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Count messages in the system
        if env.state is not None:
            messages = np.sum(env.state[env.n_obs_nghbr:-env.n_obs_nghbr] == 1)
            message_history.append(messages)
        else:
            messages = 0
            message_history.append(messages)
        
        if step % 5 == 0:
            print(f"Step {step}: Messages = {messages}, Reward = {reward:.3f}")
            if env.state is not None:
                print(f"  State: {env.state[env.n_obs_nghbr:-env.n_obs_nghbr]}")
            else:
                print("  State is not initialized.")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print(f"Message count over time: {message_history}")
    env.close()
    print("Message propagation example completed!\n")


def neighborhood_observation_example():
    """Example with different observation neighborhood sizes."""
    print("=== Neighborhood Observation Example ===")
    
    neighborhood_configs = [
        {"n_obs_neighbors": 1, "name": "1-hop neighbors (minimum)"},
        {"n_obs_neighbors": 2, "name": "2-hop neighbors"},
        {"n_obs_neighbors": 3, "name": "3-hop neighbors"},
    ]
    
    for config in neighborhood_configs:
        print(f"\nTesting: {config['name']}")
        
        # Remove 'name' from config for environment creation
        env_config = {k: v for k, v in config.items() if k != 'name'}
        env = LineMsgEnv(
            num_agents=7,
            **env_config,
            render_mode="human"
        )
        
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Neighborhood size: {env.n_obs_nghbr}")
        print(f"  Observation dimension per agent: {obs.shape[1]}")
        
        # Run for 10 steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                obs, info = env.reset()
        
        print(f"  Total reward over 10 steps: {total_reward:.3f}")
        env.close()
    
    print("Neighborhood observation example completed!\n")


def short_episode_example():
    """Example with short episodes."""
    print("=== Short Episode Example ===")
    
    # Create environment with short episodes
    env = LineMsgEnv(
        num_agents=5,
        max_iter=15,  # Short episodes
        render_mode="human",
    )
    
    print(f"Environment created with max_iter = {env.max_iter}")
    
    # Run multiple episodes
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            print(f"  Step {step_count}: Reward = {reward:.3f}")
            
            if terminated or truncated:
                print(f"  Episode finished after {step_count} steps")
                print(f"  Total episode reward: {episode_reward:.3f}")
                break
    
    env.close()
    print("Short episode example completed!\n")


def visualization_example():
    """Example demonstrating visualization capabilities."""
    print("=== Visualization Example ===")
    
    # Create environment with rgb_array rendering
    env = LineMsgEnv(
        num_agents=8,
        render_mode="rgb_array"
    )
    
    obs, info = env.reset()
    print(f"Environment created with rgb_array rendering")
    print(f"Observation shape: {obs.shape}")
    
    # Take steps and visualize
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        render_output = env.render()
        if render_output is not None:
            if isinstance(render_output, np.ndarray):
                print(f"Step {step + 1}: Render output shape = {render_output.shape}")
            else:
                print(f"Step {step + 1}: Render output type = {type(render_output)}")
            print(f"  Reward = {reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Visualization example completed!\n")


def main():
    """Run all examples."""
    print("Line Message Environment Examples")
    print("=" * 35)
    
    try:
        basic_example()
        message_propagation_example()
        neighborhood_observation_example()
        short_episode_example()
        visualization_example()
        
        print("All examples completed successfully!")
        print("\nTo test the environment more thoroughly, run:")
        print("python test_env.py")
        
    except Exception as e:
        print(f"Error during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 