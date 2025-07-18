#!/usr/bin/env python3
"""
Example script demonstrating the theoretical model concepts in the wireless communication environment.
This shows how our implementation maps to the Networked MDP framework described in the research literature.
"""

import numpy as np
from wireless_comm_env import WirelessCommEnv

def demonstrate_theoretical_mapping():
    """Demonstrate how our implementation maps to the theoretical model."""
    print("=== Wireless Communication Environment - Theoretical Model Demo ===")
    
    # Create environment with theoretical parameters
    env = WirelessCommEnv(
        grid_x=3,  # 3 users in x-direction
        grid_y=3,  # 3 users in y-direction  
        ddl=2,     # deadline horizon d_i = 2
        packet_arrival_probability=0.8,      # p_i = 0.8
        success_transmission_probability=0.8, # q_k = 0.8
        n_obs_neighbors=1,                   # neighborhood radius
        render_mode="rgb_array"
    )
    
    print(f"Theoretical Model Mapping:")
    print(f"- Users (N): {env.n_agents} agents (grid_x * grid_y = {env.grid_x} * {env.grid_y})")
    print(f"- Access Points (Y): {(env.grid_x-1) * (env.grid_y-1)} access points at intersections")
    print(f"- Deadline Horizon (d_i): {env.ddl}")
    print(f"- Packet Arrival Probability (p_i): {env.p}")
    print(f"- Transmission Success Probability (q_k): {env.q}")
    print(f"- Neighborhood Radius: {env.n_obs_nghbr}")
    
    return env

def demonstrate_state_representation():
    """Demonstrate the binary state representation."""
    print("\n=== Binary State Representation ===")
    
    env = WirelessCommEnv(grid_x=2, grid_y=2, ddl=3, render_mode="rgb_array")
    obs, info = env.reset()
    
    print(f"State shape: {env.state.shape}")
    print(f"Observation shape: {obs.shape}")
    
    # Show state for each agent
    for agent_id in range(env.n_agents):
        agent_x = agent_id // env.grid_y
        agent_y = agent_id % env.grid_y
        agent_state = env.state[:, agent_x + env.n_obs_nghbr, agent_y + env.n_obs_nghbr]
        
        print(f"\nAgent {agent_id + 1} (position {agent_x}, {agent_y}):")
        print(f"  Binary state s_{agent_id + 1} = {agent_state}")
        print(f"  Interpretation: {agent_state[0]} packets with deadline 1, {agent_state[1]} with deadline 2, {agent_state[2]} with deadline 3")
    
    env.close()

def demonstrate_access_point_subsets():
    """Demonstrate access point subsets for each user."""
    print("\n=== Access Point Subsets ===")
    
    env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="rgb_array")
    obs, info = env.reset()
    
    print("Access point subsets Y_i for each user:")
    
    for agent_id in range(env.n_agents):
        agent_x = agent_id // env.grid_y
        agent_y = agent_id % env.grid_y
        
        print(f"\nUser {agent_id + 1} (position {agent_x}, {agent_y}):")
        print(f"  Available access points Y_{agent_id + 1}:")
        
        legal_actions = []
        for action in range(5):
            ap_x, ap_y = env.access_point_mapping(agent_x, agent_y, action)
            action_names = ['null', 'UL', 'LL', 'UR', 'LR']
            
            if action == 0:
                print(f"    {action_names[action]}: no transmission")
                legal_actions.append(action_names[action])
            elif ap_x is not None and ap_y is not None:
                print(f"    {action_names[action]}: access point ({ap_x}, {ap_y})")
                legal_actions.append(action_names[action])
        
        print(f"  Legal actions A_{agent_id + 1} = {legal_actions}")
    
    env.close()

def demonstrate_conflict_graph():
    """Demonstrate the conflict graph (neighborhood structure)."""
    print("\n=== Conflict Graph (Neighborhood Structure) ===")
    
    env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="rgb_array")
    obs, info = env.reset()
    
    print("Neighborhood structure N_i (users sharing access points):")
    
    # Create access point mapping
    access_point_users = {}
    for agent_id in range(env.n_agents):
        agent_x = agent_id // env.grid_y
        agent_y = agent_id % env.grid_y
        
        for action in range(1, 5):  # Skip idle action
            ap_x, ap_y = env.access_point_mapping(agent_x, agent_y, action)
            if ap_x is not None and ap_y is not None:
                ap_key = (ap_x, ap_y)
                if ap_key not in access_point_users:
                    access_point_users[ap_key] = []
                access_point_users[ap_key].append(agent_id + 1)
    
    # Show conflicts
    for ap_key, users in access_point_users.items():
        if len(users) > 1:
            print(f"  Access point {ap_key}: Users {users} (conflict potential)")
    
    # Show neighborhoods
    for agent_id in range(env.n_agents):
        agent_x = agent_id // env.grid_y
        agent_y = agent_id % env.grid_y
        
        neighbors = set()
        for action in range(1, 5):
            ap_x, ap_y = env.access_point_mapping(agent_x, agent_y, action)
            if ap_x is not None and ap_y is not None:
                ap_key = (ap_x, ap_y)
                if ap_key in access_point_users:
                    for user in access_point_users[ap_key]:
                        if user != agent_id + 1:
                            neighbors.add(user)
        
        print(f"  N_{agent_id + 1} = {sorted(neighbors)}")
    
    env.close()

def demonstrate_transmission_dynamics():
    """Demonstrate packet transmission dynamics."""
    print("\n=== Packet Transmission Dynamics ===")
    
    env = WirelessCommEnv(grid_x=2, grid_y=2, ddl=2, render_mode="rgb_array")
    obs, info = env.reset()
    
    print("Initial state:")
    print(f"  State: {env.state[:, env.n_obs_nghbr:env.n_obs_nghbr+env.grid_x, env.n_obs_nghbr:env.n_obs_nghbr+env.grid_y]}")
    
    # Take actions that will cause conflicts
    actions = np.array([1, 1, 0, 0], dtype=int)  # First two agents try to transmit to same access point
    print(f"\nActions: {actions}")
    
    obs, reward, terminated, truncated, info = env.step(actions)
    
    print(f"Reward: {reward}")
    print(f"New state: {env.state[:, env.n_obs_nghbr:env.n_obs_nghbr+env.grid_x, env.n_obs_nghbr:env.n_obs_nghbr+env.grid_y]}")
    
    # Show what happened
    for agent_id in range(env.n_agents):
        agent_x = agent_id // env.grid_y
        agent_y = agent_id % env.grid_y
        action = actions[agent_id]
        ap_x, ap_y = env.access_point_mapping(agent_x, agent_y, action)
        
        if action == 0:
            print(f"  Agent {agent_id + 1}: Idle (no transmission)")
        elif ap_x is not None and ap_y is not None:
            print(f"  Agent {agent_id + 1}: Transmitted to access point ({ap_x}, {ap_y})")
        else:
            print(f"  Agent {agent_id + 1}: Illegal action (no valid access point)")
    
    env.close()

def demonstrate_networked_mdp_properties():
    """Demonstrate Networked MDP properties."""
    print("\n=== Networked MDP Properties ===")
    
    env = WirelessCommEnv(grid_x=2, grid_y=2, ddl=2, render_mode="rgb_array")
    obs, info = env.reset()
    
    print("Networked MDP Components:")
    print(f"1. Local State Spaces S_i: {env.observation_space.shape}")
    print(f"2. Local Action Spaces A_i: {env.action_space}")
    print(f"3. Local Transition Probabilities: Implemented in step() function")
    print(f"4. Local Reward Functions r_i(·): Implemented in step() function")
    print(f"5. Neighborhood Interactions: {env.n_obs_nghbr} neighbor radius")
    
    # Show local observation for one agent
    agent_id = 0
    local_obs = obs[agent_id]
    print(f"\nLocal observation for agent {agent_id + 1}:")
    print(f"  Shape: {local_obs.shape}")
    print(f"  Values: {local_obs}")
    
    env.close()

if __name__ == "__main__":
    print("Wireless Communication Environment - Theoretical Model Demonstration")
    print("=" * 70)
    
    # Run all demonstrations
    env = demonstrate_theoretical_mapping()
    demonstrate_state_representation()
    demonstrate_access_point_subsets()
    demonstrate_conflict_graph()
    demonstrate_transmission_dynamics()
    demonstrate_networked_mdp_properties()
    
    env.close()
    
    print("\n" + "=" * 70)
    print("Theoretical Model Demo Complete!")
    print("\nThis implementation provides:")
    print("✅ Faithful mapping to Networked MDP framework")
    print("✅ Binary state representation with deadlines")
    print("✅ Access point subsets and conflict detection")
    print("✅ Local interaction structure")
    print("✅ Configurable network topology")
    print("✅ Research-ready environment for MARL studies") 