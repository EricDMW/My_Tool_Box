#!/usr/bin/env python3
"""
Example script demonstrating the corrected action mapping in the wireless communication environment.
This shows how illegal actions are properly handled for edge and corner agents.
"""

import numpy as np
from wireless_comm_env import WirelessCommEnv

def demonstrate_action_mapping():
    """Demonstrate the action mapping for different agent positions."""
    print("=== Wireless Communication Environment - Action Mapping Demo ===")
    
    # Create a 3x3 environment to show edge and corner cases
    env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="rgb_array")
    obs, info = env.reset()
    
    print(f"Environment: {env.grid_x}x{env.grid_y} grid with {env.n_agents} agents")
    print("\nAgent positions and their legal actions:")
    
    # Test each agent position
    for agent_id in range(env.n_agents):
        agent_x = agent_id // env.grid_y
        agent_y = agent_id % env.grid_y
        
        # Determine agent type
        if (agent_x == 0 or agent_x == env.grid_x - 1) and (agent_y == 0 or agent_y == env.grid_y - 1):
            agent_type = "Corner"
        elif agent_x == 0 or agent_x == env.grid_x - 1 or agent_y == 0 or agent_y == env.grid_y - 1:
            agent_type = "Edge"
        else:
            agent_type = "Center"
        
        print(f"\nAgent {agent_id + 1} ({agent_type}) at position ({agent_x}, {agent_y}):")
        
        # Test each action
        legal_actions = []
        for action in range(5):
            ap_x, ap_y = env.access_point_mapping(agent_x, agent_y, action)
            action_names = ['Idle', 'UL', 'LL', 'UR', 'LR']
            
            if action == 0:  # Idle is always legal
                legal_actions.append(action)
                print(f"  Action {action} ({action_names[action]}): LEGAL - No transmission")
            elif ap_x is not None and ap_y is not None:
                legal_actions.append(action)
                print(f"  Action {action} ({action_names[action]}): LEGAL -> Access Point ({ap_x}, {ap_y})")
            else:
                print(f"  Action {action} ({action_names[action]}): ILLEGAL - No valid access point")
        
        print(f"  Legal actions: {legal_actions} ({len(legal_actions)} total)")
    
    env.close()

def demonstrate_rendering_with_actions():
    """Demonstrate rendering with different actions."""
    print("\n=== Rendering Demo with Actions ===")
    
    env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set some actions (including legal and illegal ones)
    actions = np.array([0, 1, 2, 3, 4, 0, 0, 0, 0], dtype=int)
    
    print("Setting actions for first 5 agents:", actions[:5])
    print("Note: Some actions may be illegal for edge agents")
    
    obs, reward, terminated, truncated, info = env.step(actions)
    
    # Render the result
    frame = env.render()
    print(f"Rendered frame shape: {frame.shape}")
    
    # Save the frame
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(frame)
    plt.axis('off')
    plt.title('Action Mapping Demo - Blue cubes are agents, arrows show transmission attempts')
    plt.savefig('action_mapping_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved demo image as 'action_mapping_demo.png'")
    
    env.close()

def demonstrate_legal_vs_illegal():
    """Demonstrate the difference between legal and illegal actions."""
    print("\n=== Legal vs Illegal Actions Demo ===")
    
    env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="rgb_array")
    obs, info = env.reset()
    
    # Test corner agent (Agent 1 at position 0,0)
    agent_id = 0
    agent_x = 0
    agent_y = 0
    
    print(f"Testing corner agent {agent_id + 1} at position ({agent_x}, {agent_y}):")
    
    # Test illegal action (UL - should fail)
    actions_illegal = np.zeros(env.n_agents, dtype=int)
    actions_illegal[agent_id] = 1  # UL action
    obs, reward, terminated, truncated, info = env.step(actions_illegal)
    
    ap_x, ap_y = env.access_point_mapping(agent_x, agent_y, 1)
    print(f"  Illegal action 1 (UL): Access point = ({ap_x}, {ap_y})")
    print(f"  Reward: {reward}")
    
    # Reset and test legal action (LR - should work)
    obs, info = env.reset()
    actions_legal = np.zeros(env.n_agents, dtype=int)
    actions_legal[agent_id] = 4  # LR action
    obs, reward, terminated, truncated, info = env.step(actions_legal)
    
    ap_x, ap_y = env.access_point_mapping(agent_x, agent_y, 4)
    print(f"  Legal action 4 (LR): Access point = ({ap_x}, {ap_y})")
    print(f"  Reward: {reward}")
    
    env.close()

if __name__ == "__main__":
    demonstrate_action_mapping()
    demonstrate_rendering_with_actions()
    demonstrate_legal_vs_illegal()
    
    print("\n=== Demo Complete ===")
    print("The environment now correctly handles:")
    print("- Legal actions for all agent positions")
    print("- Illegal actions that point to invalid access points")
    print("- Proper rendering of transmission attempts")
    print("- Automatic filtering of invalid actions") 