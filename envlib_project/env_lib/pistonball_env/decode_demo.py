#!/usr/bin/env python3
"""
Interactive demonstration of discrete action decoding in the pistonball environment.

This script shows how single integer actions are decoded into individual piston movements.
"""

import numpy as np

def decode_discrete_action(n_pistons, action):
    """
    Decode a discrete action into continuous piston actions.
    
    Args:
        n_pistons: Number of pistons
        action: Discrete action integer
        
    Returns:
        Array of continuous actions for each piston
    """
    action_array = np.zeros(n_pistons)
    for i in range(n_pistons):
        piston_action = (action // (3 ** i)) % 3
        action_array[i] = piston_action - 1  # Convert 0,1,2 to -1,0,1
    return action_array

def encode_continuous_action(continuous_actions):
    """
    Encode continuous actions back to discrete action.
    
    Args:
        continuous_actions: Array of continuous actions (-1, 0, 1)
        
    Returns:
        Discrete action integer
    """
    n_pistons = len(continuous_actions)
    discrete_action = 0
    for i in range(n_pistons):
        piston_action = int(continuous_actions[i] + 1)  # Convert -1,0,1 to 0,1,2
        discrete_action += piston_action * (3 ** i)
    return discrete_action

def print_action_mapping(n_pistons, action):
    """Print detailed mapping of discrete action to continuous actions."""
    continuous_actions = decode_discrete_action(n_pistons, action)
    
    print(f"\n🔍 Decoding Action {action} for {n_pistons} pistons:")
    print("=" * 50)
    
    # Show decoding process
    for i in range(n_pistons):
        piston_action = (action // (3 ** i)) % 3
        continuous_action = piston_action - 1
        action_name = {0: "DOWN", 1: "STAY", 2: "UP"}[piston_action]
        print(f"Piston {i}: ({action} // {3**i}) % 3 = {piston_action} -> {continuous_action} ({action_name})")
    
    print(f"\n📊 Result: {continuous_actions}")
    
    # Show visual representation
    print("\n🎮 Visual Representation:")
    for i, action_val in enumerate(continuous_actions):
        if action_val == 1:
            symbol = "⬆️"
        elif action_val == -1:
            symbol = "⬇️"
        else:
            symbol = "⏸️"
        print(f"Piston {i}: {symbol} ({action_val:2.0f})")
    
    return continuous_actions

def demo_basic_examples():
    """Demonstrate basic decoding examples."""
    print("🎯 Basic Discrete Action Decoding Examples")
    print("=" * 60)
    
    n_pistons = 3
    max_action = 3 ** n_pistons - 1
    
    print(f"Environment: {n_pistons} pistons")
    print(f"Action space: 0 to {max_action} ({3**n_pistons} total actions)")
    print(f"Each piston: 0=DOWN, 1=STAY, 2=UP")
    
    # Test key actions
    key_actions = [0, 13, 26, 5, 22]
    
    for action in key_actions:
        print_action_mapping(n_pistons, action)
        print()

def demo_all_actions_small():
    """Show all possible actions for a small number of pistons."""
    print("📋 All Possible Actions for 2 Pistons")
    print("=" * 50)
    
    n_pistons = 2
    max_action = 3 ** n_pistons - 1
    
    print(f"Action space: 0 to {max_action} ({3**n_pistons} total actions)")
    print()
    
    print("| Action | Piston 0 | Piston 1 | Continuous | Visual |")
    print("|--------|----------|----------|------------|--------|")
    
    for action in range(3 ** n_pistons):
        continuous = decode_discrete_action(n_pistons, action)
        
        # Create visual representation
        visual = ""
        for val in continuous:
            if val == 1:
                visual += "⬆️"
            elif val == -1:
                visual += "⬇️"
            else:
                visual += "⏸️"
        
        # Get action names
        piston0_action = (action // 1) % 3
        piston1_action = (action // 3) % 3
        
        print(f"| {action:6d} | {piston0_action:8d} | {piston1_action:8d} | {str(continuous):10s} | {visual:6s} |")

def demo_encoding_decoding():
    """Demonstrate encoding and decoding round-trip."""
    print("🔄 Encoding and Decoding Round-trip")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        np.array([1, -1, 0]),    # up, down, stay
        np.array([0, 0, 0]),     # all stay
        np.array([1, 1, 1]),     # all up
        np.array([-1, -1, -1]),  # all down
        np.array([1, 0, -1]),    # up, stay, down
    ]
    
    for i, continuous_actions in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {continuous_actions}")
        
        # Encode
        discrete_action = encode_continuous_action(continuous_actions)
        print(f"Encoded to: {discrete_action}")
        
        # Decode
        decoded_actions = decode_discrete_action(len(continuous_actions), discrete_action)
        print(f"Decoded to: {decoded_actions}")
        
        # Verify
        if np.array_equal(continuous_actions, decoded_actions):
            print("✅ Round-trip successful!")
        else:
            print("❌ Round-trip failed!")

def interactive_demo():
    """Interactive demo where user can input actions."""
    print("🎮 Interactive Discrete Action Decoder")
    print("=" * 50)
    
    while True:
        try:
            # Get input
            n_pistons = int(input("\nEnter number of pistons (1-5, or 0 to exit): "))
            if n_pistons == 0:
                break
            if n_pistons < 1 or n_pistons > 5:
                print("Please enter a number between 1 and 5.")
                continue
            
            max_action = 3 ** n_pistons - 1
            action = int(input(f"Enter discrete action (0-{max_action}): "))
            
            if action < 0 or action > max_action:
                print(f"Please enter a number between 0 and {max_action}.")
                continue
            
            # Decode and display
            print_action_mapping(n_pistons, action)
            
        except ValueError:
            print("Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def demo_large_teams():
    """Show how action space grows with team size."""
    print("📈 Action Space Growth with Team Size")
    print("=" * 50)
    
    print("| Pistons | Total Actions | Max Action | Memory Usage |")
    print("|---------|---------------|------------|--------------|")
    
    for n_pistons in range(1, 11):
        total_actions = 3 ** n_pistons
        max_action = total_actions - 1
        memory_mb = (total_actions * 4) / (1024 * 1024)  # Assuming 4 bytes per action
        
        print(f"| {n_pistons:7d} | {total_actions:13d} | {max_action:10d} | {memory_mb:10.2f} MB |")
        
        if total_actions > 1000000:
            print(f"⚠️  {n_pistons} pistons: Action space too large for practical use!")

def main():
    """Run all demonstrations."""
    print("🎯 Discrete Action Decoding Demonstration")
    print("=" * 60)
    print("This demo shows how discrete actions are decoded in the pistonball environment.")
    print("Each discrete action is converted to individual piston movements.")
    
    try:
        # Run demonstrations
        demo_basic_examples()
        demo_all_actions_small()
        demo_encoding_decoding()
        demo_large_teams()
        
        # Interactive demo
        print("\n" + "=" * 60)
        interactive_demo()
        
        print("\n🎉 Demonstration completed!")
        print("\nKey Takeaways:")
        print("• Discrete actions use base-3 encoding")
        print("• Each piston has 3 actions: 0=DOWN, 1=STAY, 2=UP")
        print("• Action space grows exponentially: 3^n_pistons")
        print("• Decoding uses integer division and modulo operations")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 