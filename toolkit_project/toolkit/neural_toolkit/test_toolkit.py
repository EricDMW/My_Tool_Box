#!/usr/bin/env python3
"""
Test script for Neural Toolkit

This script tests all major components of the neural toolkit to ensure they work correctly.
"""

import torch
import numpy as np
from toolkit.neural_toolkit import (
    # Policy Networks
    PolicyFactory,
    
    # Value Networks
    ValueFactory,
    
    # Q Networks
    QNetworkFactory,
    
    # Encoders and Decoders
    EncoderFactory,
    DecoderFactory,
    
    # Discrete Tools
    DiscreteTools,
    
    # Utilities
    NetworkUtils
)


def test_policy_networks():
    """Test policy network creation and forward pass"""
    print("Testing Policy Networks...")
    
    # Test MLP Policy
    mlp_policy = PolicyFactory.create_policy(
        policy_type='mlp',
        input_dim=10,
        output_dim=4,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(32, 10)
    output = mlp_policy(x)
    assert output.shape == (32, 4), f"MLP Policy output shape: {output.shape}"
    print("✓ MLP Policy Network")
    
    # Test CNN Policy
    cnn_policy = PolicyFactory.create_policy(
        policy_type='cnn',
        input_channels=3,
        output_dim=4,
        conv_dims=[32, 64, 128],
        fc_dims=[256, 128]
    )
    
    # Test forward pass
    x = torch.randn(32, 3, 64, 64)
    output = cnn_policy(x)
    assert output.shape == (32, 4), f"CNN Policy output shape: {output.shape}"
    print("✓ CNN Policy Network")
    
    # Test RNN Policy
    rnn_policy = PolicyFactory.create_policy(
        policy_type='rnn',
        input_dim=10,
        output_dim=4,
        hidden_dim=256,
        num_layers=2,
        rnn_type='lstm'
    )
    
    # Test forward pass
    x = torch.randn(32, 20, 10)  # (batch, seq_len, input_dim)
    output, hidden = rnn_policy(x)
    assert output.shape == (32, 4), f"RNN Policy output shape: {output.shape}"
    print("✓ RNN Policy Network")
    
    print("All Policy Networks tested successfully!\n")


def test_value_networks():
    """Test value network creation and forward pass"""
    print("Testing Value Networks...")
    
    # Test MLP Value Network
    mlp_value = ValueFactory.create_value_network(
        network_type='mlp',
        input_dim=10,
        output_dim=1,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(32, 10)
    output = mlp_value(x)
    assert output.shape == (32, 1), f"MLP Value output shape: {output.shape}"
    print("✓ MLP Value Network")
    
    # Test CNN Value Network
    cnn_value = ValueFactory.create_value_network(
        network_type='cnn',
        input_channels=3,
        output_dim=1,
        conv_dims=[32, 64, 128],
        fc_dims=[256, 128]
    )
    
    # Test forward pass
    x = torch.randn(32, 3, 64, 64)
    output = cnn_value(x)
    assert output.shape == (32, 1), f"CNN Value output shape: {output.shape}"
    print("✓ CNN Value Network")
    
    print("All Value Networks tested successfully!\n")


def test_q_networks():
    """Test Q-network creation and forward pass"""
    print("Testing Q-Networks...")
    
    # Test MLP Q-Network
    mlp_q = QNetworkFactory.create_q_network(
        network_type='mlp',
        state_dim=10,
        action_dim=4,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Test forward pass
    state = torch.randn(32, 10)
    output = mlp_q(state)
    assert output.shape == (32, 4), f"MLP Q-Network output shape: {output.shape}"
    print("✓ MLP Q-Network")
    
    # Test Dueling Q-Network
    dueling_q = QNetworkFactory.create_q_network(
        network_type='dueling',
        state_dim=10,
        action_dim=4,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Test forward pass
    state = torch.randn(32, 10)
    output = dueling_q(state)
    assert output.shape == (32, 4), f"Dueling Q-Network output shape: {output.shape}"
    print("✓ Dueling Q-Network")
    
    print("All Q-Networks tested successfully!\n")


def test_encoders_decoders():
    """Test encoder and decoder creation and forward pass"""
    print("Testing Encoders and Decoders...")
    
    # Test MLP Encoder
    mlp_encoder = EncoderFactory.create_encoder(
        encoder_type='mlp',
        input_dim=100,
        latent_dim=32,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(32, 100)
    latent = mlp_encoder(x)
    assert latent.shape == (32, 32), f"MLP Encoder output shape: {latent.shape}"
    print("✓ MLP Encoder")
    
    # Test MLP Decoder
    mlp_decoder = DecoderFactory.create_decoder(
        decoder_type='mlp',
        latent_dim=32,
        output_dim=100,
        hidden_dims=[128, 256],
        activation='relu',
        dropout=0.1
    )
    
    # Test forward pass
    z = torch.randn(32, 32)
    output = mlp_decoder(z)
    assert output.shape == (32, 100), f"MLP Decoder output shape: {output.shape}"
    print("✓ MLP Decoder")
    
    # Test VAE Encoder
    vae_encoder = EncoderFactory.create_encoder(
        encoder_type='vae',
        input_dim=100,
        latent_dim=32,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(32, 100)
    z, mu, logvar = vae_encoder(x)
    assert z.shape == (32, 32), f"VAE Encoder z shape: {z.shape}"
    assert mu.shape == (32, 32), f"VAE Encoder mu shape: {mu.shape}"
    assert logvar.shape == (32, 32), f"VAE Encoder logvar shape: {logvar.shape}"
    print("✓ VAE Encoder")
    
    print("All Encoders and Decoders tested successfully!\n")


def test_discrete_tools():
    """Test discrete tools functionality"""
    print("Testing Discrete Tools...")
    
    # Create tables
    state_size = 10
    action_size = 4
    
    q_table = DiscreteTools.create_q_table(state_size, action_size)
    value_table = DiscreteTools.create_value_table(state_size)
    policy_table = DiscreteTools.create_policy_table(state_size, action_size)
    
    # Test Q-table operations
    q_table.set_value(0, 1.0, 1)
    value = q_table.get_value(0, 1)
    assert value == 1.0, f"Q-table value: {value}"
    
    # Test Q-learning update
    DiscreteTools.q_learning_update(q_table, state=0, action=1, reward=1.0, 
                                   next_state=2, gamma=0.9, alpha=0.1)
    
    # Test epsilon-greedy policy
    action = DiscreteTools.epsilon_greedy_policy(q_table, state=0, epsilon=0.1)
    assert 0 <= action < action_size, f"Epsilon-greedy action: {action}"
    
    # Test softmax policy
    action = DiscreteTools.softmax_policy(q_table, state=0, temperature=1.0)
    assert 0 <= action < action_size, f"Softmax action: {action}"
    
    print("✓ Q-Table Operations")
    print("✓ Q-Learning Update")
    print("✓ Exploration Policies")
    print("All Discrete Tools tested successfully!\n")


def test_network_utils():
    """Test network utilities"""
    print("Testing Network Utilities...")
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 4)
    )
    
    # Test weight initialization
    NetworkUtils.initialize_weights(model, method='xavier_uniform')
    print("✓ Weight Initialization")
    
    # Test parameter counting
    total_params = NetworkUtils.count_parameters(model)
    assert total_params > 0, f"Parameter count: {total_params}"
    print("✓ Parameter Counting")
    
    # Test optimizer creation
    optimizer = NetworkUtils.create_optimizer(
        model=model,
        optimizer_type='adam',
        lr=1e-3,
        weight_decay=1e-4
    )
    assert optimizer is not None, "Optimizer creation failed"
    print("✓ Optimizer Creation")
    
    # Test scheduler creation
    scheduler = NetworkUtils.create_scheduler(
        optimizer=optimizer,
        scheduler_type='cosine',
        T_max=1000
    )
    assert scheduler is not None, "Scheduler creation failed"
    print("✓ Scheduler Creation")
    
    # Test MLP layer creation
    mlp_layers = NetworkUtils.create_mlp_layers(
        input_dim=10,
        hidden_dims=[256, 128],
        output_dim=4,
        activation='relu',
        dropout=0.1,
        layer_norm=True
    )
    
    x = torch.randn(32, 10)
    output = mlp_layers(x)
    assert output.shape == (32, 4), f"MLP layers output shape: {output.shape}"
    print("✓ MLP Layer Creation")
    
    print("All Network Utilities tested successfully!\n")


def test_integration():
    """Test integration of multiple components"""
    print("Testing Integration...")
    
    # Create a complete RL agent setup
    state_dim = 10
    action_dim = 4
    
    # Policy network
    policy = PolicyFactory.create_policy(
        policy_type='mlp',
        input_dim=state_dim,
        output_dim=action_dim,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Value network
    value = ValueFactory.create_value_network(
        network_type='mlp',
        input_dim=state_dim,
        output_dim=1,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Initialize weights
    NetworkUtils.initialize_weights(policy, method='xavier_uniform')
    NetworkUtils.initialize_weights(value, method='xavier_uniform')
    
    # Create optimizers
    policy_optimizer = NetworkUtils.create_optimizer(
        policy, optimizer_type='adam', lr=3e-4
    )
    value_optimizer = NetworkUtils.create_optimizer(
        value, optimizer_type='adam', lr=1e-3
    )
    
    # Test forward passes
    state = torch.randn(32, state_dim)
    policy_output = policy(state)
    value_output = value(state)
    
    assert policy_output.shape == (32, action_dim), f"Policy output shape: {policy_output.shape}"
    assert value_output.shape == (32, 1), f"Value output shape: {value_output.shape}"
    
    print("✓ Complete RL Agent Setup")
    print("✓ Forward Passes")
    print("✓ Optimizer Integration")
    
    print("Integration test completed successfully!\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Neural Toolkit Test Suite")
    print("=" * 60)
    
    try:
        test_policy_networks()
        test_value_networks()
        test_q_networks()
        test_encoders_decoders()
        test_discrete_tools()
        test_network_utils()
        test_integration()
        
        print("=" * 60)
        print("🎉 All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 