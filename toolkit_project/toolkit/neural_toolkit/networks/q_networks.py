"""
Q-Value Networks for Reinforcement Learning

This module provides flexible Q-value network implementations for action-value estimation.
Supports various architectures including MLP, CNN, RNN, and Transformer-based Q-networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod


class BaseQNetwork(nn.Module, ABC):
    """Base class for all Q-value networks"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 device: str = 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
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
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        pass


class MLPQNetwork(BaseQNetwork):
    """Multi-Layer Perceptron Q-Network"""
    
    def _build_network(self):
        # State processing layers
        state_layers = []
        prev_dim = self.state_dim
        
        for hidden_dim in self.hidden_dims:
            state_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.layer_norm:
                state_layers.append(nn.LayerNorm(hidden_dim))
            if self.dropout > 0:
                state_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.state_layers = nn.Sequential(*state_layers)
        
        # Q-value output layer
        self.q_layer = nn.Linear(prev_dim, self.action_dim)
        
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        state_features = self.state_layers(state)
        state_features = self._get_activation()(state_features)
        q_values = self.q_layer(state_features)
        return q_values


class DuelingQNetwork(BaseQNetwork):
    """Dueling Q-Network with separate value and advantage streams"""
    
    def _build_network(self):
        # Shared feature layers
        shared_layers = []
        prev_dim = self.state_dim
        
        for hidden_dim in self.hidden_dims[:-1]:  # Use all but last hidden dim for shared
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.layer_norm:
                shared_layers.append(nn.LayerNorm(hidden_dim))
            if self.dropout > 0:
                shared_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_layers = nn.Sequential(
            nn.Linear(prev_dim, self.hidden_dims[-1]),
            nn.Linear(self.hidden_dims[-1], 1)
        )
        
        # Advantage stream
        self.advantage_layers = nn.Sequential(
            nn.Linear(prev_dim, self.hidden_dims[-1]),
            nn.Linear(self.hidden_dims[-1], self.action_dim)
        )
        
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        shared_features = self.shared_layers(state)
        shared_features = self._get_activation()(shared_features)
        
        value = self.value_layers(shared_features)
        advantage = self.advantage_layers(shared_features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class CNQNetwork(BaseQNetwork):
    """Convolutional Neural Network Q-Network for image-based observations"""
    
    def __init__(self,
                 input_channels: int,
                 action_dim: int,
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
        
        super().__init__(state_dim=0, action_dim=action_dim, 
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
        self.q_layer = nn.Linear(prev_dim, self.action_dim)
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        # state shape: (batch_size, channels, height, width)
        conv_features = self.conv_layers(state)
        pooled = self.adaptive_pool(conv_features)
        flattened = pooled.view(pooled.size(0), -1)
        
        fc_features = self.fc_layers(flattened)
        fc_features = self._get_activation()(fc_features)
        
        q_values = self.q_layer(fc_features)
        return q_values


class RNQNetwork(BaseQNetwork):
    """Recurrent Neural Network Q-Network for sequential observations"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
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
        
        super().__init__(state_dim=state_dim, action_dim=action_dim,
                        hidden_dims=fc_dims, activation=activation,
                        dropout=dropout, layer_norm=layer_norm, device=device)
    
    def _build_network(self):
        # Build RNN layer
        if self.rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                self.state_dim, self.hidden_dim, self.num_layers,
                bidirectional=self.bidirectional, dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                self.state_dim, self.hidden_dim, self.num_layers,
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
        self.q_layer = nn.Linear(prev_dim, self.action_dim)
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None, 
                hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # state shape: (batch_size, seq_len, state_dim)
        rnn_out, hidden = self.rnn(state, hidden)
        
        # Use the last output
        last_output = rnn_out[:, -1, :]
        
        fc_features = self.fc_layers(last_output)
        fc_features = self._get_activation()(fc_features)
        
        q_values = self.q_layer(fc_features)
        return q_values, hidden


class TransformerQNetwork(BaseQNetwork):
    """Transformer-based Q-Network for complex sequential patterns"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
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
        
        super().__init__(state_dim=state_dim, action_dim=action_dim,
                        hidden_dims=fc_dims, activation=activation,
                        dropout=dropout, layer_norm=layer_norm, device=device)
    
    def _build_network(self):
        # Input projection
        self.input_projection = nn.Linear(self.state_dim, self.d_model)
        
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
        self.q_layer = nn.Linear(prev_dim, self.action_dim)
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # state shape: (batch_size, seq_len, state_dim)
        x = self.input_projection(state)
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
        
        q_values = self.q_layer(fc_features)
        return q_values


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


class QNetworkFactory:
    """Factory class for creating Q-networks"""
    
    @staticmethod
    def create_q_network(network_type: str, **kwargs) -> BaseQNetwork:
        """Create a Q-network based on type"""
        network_types = {
            'mlp': MLPQNetwork,
            'dueling': DuelingQNetwork,
            'cnn': CNQNetwork,
            'rnn': RNQNetwork,
            'transformer': TransformerQNetwork
        }
        
        if network_type not in network_types:
            raise ValueError(f"Unsupported network type: {network_type}")
        
        return network_types[network_type](**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test MLP Q-Network
    mlp_q = QNetworkFactory.create_q_network(
        network_type='mlp',
        state_dim=10,
        action_dim=4,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Test Dueling Q-Network
    dueling_q = QNetworkFactory.create_q_network(
        network_type='dueling',
        state_dim=10,
        action_dim=4,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Test CNN Q-Network
    cnn_q = QNetworkFactory.create_q_network(
        network_type='cnn',
        input_channels=3,
        action_dim=4,
        conv_dims=[32, 64, 128],
        fc_dims=[256, 128]
    )
    
    # Test RNN Q-Network
    rnn_q = QNetworkFactory.create_q_network(
        network_type='rnn',
        state_dim=10,
        action_dim=4,
        hidden_dim=256,
        num_layers=2,
        rnn_type='lstm'
    )
    
    print("All Q-networks created successfully!") 