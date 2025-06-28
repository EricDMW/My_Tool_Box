#!/usr/bin/env python3
"""
Test script to verify that the Kuramoto environment is properly registered
and can be used with gymnasium.
"""

import gymnasium as gym
from gymnasium.envs.registration import registry

def test_environment_registration():
    """Test that all Kuramoto environments are properly registered."""
    print("Testing Kuramoto environment registration...")
    
    # List of expected environment IDs
    expected_envs = [
        'KuramotoOscillator-v0',
        'KuramotoOscillator-v1',
        'KuramotoOscillator-Constant-v0',
        'KuramotoOscillator-FreqSync-Constant-v0',
        'KuramotoOscillatorTorch-v0',
        'KuramotoOscillatorTorch-v1',
        'KuramotoOscillatorTorch-v2',
        'KuramotoOscillatorTorch-Constant-v0',
        'KuramotoOscillatorTorch-FreqSync-Constant-v0',
    ]
    
    # Check if environments are registered
    registered_envs = []
    for env_id in expected_envs:
        if env_id in registry:
            registered_envs.append(env_id)
            print(f"✓ {env_id} is registered")
        else:
            print(f"✗ {env_id} is NOT registered")
    
    print(f"\nRegistered {len(registered_envs)} out of {len(expected_envs)} environments")
    
    return len(registered_envs) == len(expected_envs)

def test_environment_creation():
    """Test that environments can be created and used."""
    print("\nTesting environment creation...")
    
    try:
        # Test NumPy version
        env = gym.make('KuramotoOscillator-v0')
        obs, info = env.reset()
        print(f"✓ KuramotoOscillator-v0 created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Initial observation shape: {obs.shape}")
        env.close()
        
        # Test PyTorch version
        env = gym.make('KuramotoOscillatorTorch-v0')
        obs, info = env.reset()
        print(f"✓ KuramotoOscillatorTorch-v0 created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Initial observation shape: {obs.shape}")
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        return False

def test_basic_interaction():
    """Test basic environment interaction."""
    print("\nTesting basic environment interaction...")
    
    try:
        env = gym.make('KuramotoOscillator-v0')
        obs, info = env.reset()
        
        # Take a few random actions
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Step {step}: reward = {reward:.4f}, done = {done}")
            
            if done:
                break
        
        env.close()
        print("✓ Basic interaction test passed")
        return True
        
    except Exception as e:
        print(f"✗ Error during interaction test: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Kuramoto Environment Setup Test")
    print("=" * 50)
    
    # Run tests
    registration_ok = test_environment_registration()
    creation_ok = test_environment_creation()
    interaction_ok = test_basic_interaction()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Registration: {'✓ PASS' if registration_ok else '✗ FAIL'}")
    print(f"Creation: {'✓ PASS' if creation_ok else '✗ FAIL'}")
    print(f"Interaction: {'✓ PASS' if interaction_ok else '✗ FAIL'}")
    
    if all([registration_ok, creation_ok, interaction_ok]):
        print("\n🎉 All tests passed! The environment is properly set up.")
    else:
        print("\n❌ Some tests failed. Please check the installation.")
    print("=" * 50) 