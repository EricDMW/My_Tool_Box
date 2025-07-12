"""
Policy Networks for Reinforcement Learning

This module provides flexible policy network implementations for both discrete and continuous action spaces.
Supports various architectures including MLP, CNN, RNN, and Transformer-based policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod


class BasePolicyNetwork(nn.Module, ABC):
    """Base class for all policy networks"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.device = device
        
        self._build_network()
    
    def _get_activation(self) -> Callable:
        """Get activation function"""
        activations = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'swish': lambda x: x * torch.sigmoid(x),
            'gelu': F.gelu
        }
        return activations.get(self.activation, F.relu)
    
    @abstractmethod
    def _build_network(self):
        """Build the network architecture"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass


class MLPPolicyNetwork(BasePolicyNetwork):
    """Multi-Layer Perceptron Policy Network"""
    
    def _build_network(self):
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)
        features = self._get_activation()(features)
        return self.output_layer(features)


class CNPolicyNetwork(BasePolicyNetwork):
    """Convolutional Neural Network Policy for image-based observations"""
    
    def __init__(self,
                 input_channels: int,
                 output_dim: int,
                 conv_dims: List[int] = [32, 64, 128],
                 fc_dims: List[int] = [256, 256],
                 kernel_sizes: List[int] = [3, 3, 3],
                 strides: List[int] = [1, 1, 1],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 device: str = 'cpu'):
        
        self.input_channels = input_channels
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        
        super().__init__(input_dim=0, output_dim=output_dim, 
                        hidden_dims=fc_dims, activation=activation,
                        dropout=dropout, layer_norm=layer_norm, device=device)
    
    def _build_network(self):
        # Build convolutional layers
        conv_layers = []
        prev_channels = self.input_channels
        
        for i, (out_channels, kernel_size, stride) in enumerate(
            zip(self.conv_dims, self.kernel_sizes, self.strides)):
            conv_layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels) if self.layer_norm else nn.Identity(),
                nn.ReLU() if self.activation == 'relu' else self._get_activation(),
                nn.Dropout2d(self.dropout) if self.dropout > 0 else nn.Identity()
            ])
            prev_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Build fully connected layers
        fc_layers = []
        prev_dim = self.conv_dims[-1]
        
        for hidden_dim in self.fc_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.layer_norm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            if self.dropout > 0:
                fc_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, channels, height, width)
        conv_features = self.conv_layers(x)
        pooled = self.adaptive_pool(conv_features)
        flattened = pooled.view(pooled.size(0), -1)
        
        fc_features = self.fc_layers(flattened)
        fc_features = self._get_activation()(fc_features)
        
        return self.output_layer(fc_features)


class RNPolicyNetwork(BasePolicyNetwork):
    """Recurrent Neural Network Policy for sequential observations"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 rnn_type: str = 'lstm',
                 bidirectional: bool = False,
                 fc_dims: List[int] = [256],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 device: str = 'cpu'):
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.fc_dims = fc_dims
        
        super().__init__(input_dim=input_dim, output_dim=output_dim,
                        hidden_dims=fc_dims, activation=activation,
                        dropout=dropout, layer_norm=layer_norm, device=device)
    
    def _build_network(self):
        # Build RNN layer
        if self.rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                self.input_dim, self.hidden_dim, self.num_layers,
                bidirectional=self.bidirectional, dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                self.input_dim, self.hidden_dim, self.num_layers,
                bidirectional=self.bidirectional, dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
        
        # Calculate RNN output dimension
        rnn_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        
        # Build fully connected layers
        fc_layers = []
        prev_dim = rnn_output_dim
        
        for hidden_dim in self.fc_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.layer_norm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            if self.dropout > 0:
                fc_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # x shape: (batch_size, seq_len, input_dim)
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Use the last output
        last_output = rnn_out[:, -1, :]
        
        fc_features = self.fc_layers(last_output)
        fc_features = self._get_activation()(fc_features)
        
        output = self.output_layer(fc_features)
        return output, hidden


class TransformerPolicyNetwork(BasePolicyNetwork):
    """Transformer-based Policy Network for complex sequential patterns"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 fc_dims: List[int] = [256],
                 activation: str = 'relu',
                 layer_norm: bool = True,
                 device: str = 'cpu'):
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.fc_dims = fc_dims
        
        super().__init__(input_dim=input_dim, output_dim=output_dim,
                        hidden_dims=fc_dims, activation=activation,
                        dropout=dropout, layer_norm=layer_norm, device=device)
    
    def _build_network(self):
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
        # Output layers
        fc_layers = []
        prev_dim = self.d_model
        
        for hidden_dim in self.fc_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.layer_norm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            if self.dropout > 0:
                fc_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        if mask is not None:
            transformer_out = self.transformer(x, src_key_padding_mask=mask)
        else:
            transformer_out = self.transformer(x)
        
        # Use the last output or mean pooling
        if mask is not None:
            # Mean pooling over non-masked positions
            mask_expanded = mask.unsqueeze(-1).expand_as(transformer_out)
            transformer_out = transformer_out.masked_fill(mask_expanded, 0)
            seq_lengths = (~mask).sum(dim=1, keepdim=True).float()
            pooled = transformer_out.sum(dim=1) / seq_lengths
        else:
            pooled = transformer_out.mean(dim=1)
        
        fc_features = self.fc_layers(pooled)
        fc_features = self._get_activation()(fc_features)
        
        return self.output_layer(fc_features)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer
    
    Note: The max_len parameter is internal and pre-computes positional encodings
    up to that length. It doesn't need to be exposed as a parameter to transformer
    networks since they can handle variable sequence lengths dynamically.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe_tensor = getattr(self, 'pe')
        x = x + pe_tensor[:x.size(0), :]
        return self.dropout(x)


class PolicyFactory:
    """Factory class for creating policy networks"""
    
    @staticmethod
    def create_policy(policy_type: str, **kwargs) -> BasePolicyNetwork:
        """Create a policy network based on type"""
        policy_types = {
            'mlp': MLPPolicyNetwork,
            'cnn': CNPolicyNetwork,
            'rnn': RNPolicyNetwork,
            'transformer': TransformerPolicyNetwork
        }
        
        if policy_type not in policy_types:
            raise ValueError(f"Unsupported policy type: {policy_type}")
        
        return policy_types[policy_type](**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test MLP Policy
    mlp_policy = PolicyFactory.create_policy(
        policy_type='mlp',
        input_dim=10,
        output_dim=4,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Test CNN Policy
    cnn_policy = PolicyFactory.create_policy(
        policy_type='cnn',
        input_channels=3,
        output_dim=4,
        conv_dims=[32, 64, 128],
        fc_dims=[256, 128]
    )
    
    # Test RNN Policy
    rnn_policy = PolicyFactory.create_policy(
        policy_type='rnn',
        input_dim=10,
        output_dim=4,
        hidden_dim=256,
        num_layers=2,
        rnn_type='lstm'
    )
    
    print("All policy networks created successfully!") 