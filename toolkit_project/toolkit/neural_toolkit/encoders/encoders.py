"""
Encoders for Neural Networks

This module provides flexible encoder implementations for feature extraction and representation learning.
Supports various architectures including MLP, CNN, RNN, Transformer, and Autoencoder encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod


class BaseEncoder(nn.Module, ABC):
    """Base class for all encoders"""
    
    def __init__(self, 
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.device = device
        
        self._build_encoder()
    
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
    def _build_encoder(self):
        """Build the encoder architecture"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass


class MLPEncoder(BaseEncoder):
    """Multi-Layer Perceptron Encoder"""
    
    def _build_encoder(self):
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
        self.latent_layer = nn.Linear(prev_dim, self.latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)
        features = self._get_activation()(features)
        latent = self.latent_layer(features)
        return latent


class CNNEncoder(BaseEncoder):
    """Convolutional Neural Network Encoder for image-based inputs"""
    
    def __init__(self,
                 input_channels: int,
                 latent_dim: int,
                 conv_dims: List[int] = [32, 64, 128, 256],
                 fc_dims: List[int] = [512, 256],
                 kernel_sizes: List[int] = [3, 3, 3, 3],
                 strides: List[int] = [1, 2, 2, 2],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 device: str = 'cpu'):
        
        self.input_channels = input_channels
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        
        super().__init__(input_dim=0, latent_dim=latent_dim, 
                        hidden_dims=fc_dims, activation=activation,
                        dropout=dropout, layer_norm=layer_norm, device=device)
    
    def _build_encoder(self):
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
        
        # Build fully connected layers
        fc_layers = []
        prev_dim = self.conv_dims[-1] * 4 * 4  # Assuming 64x64 input -> 4x4 after convs
        
        for hidden_dim in self.fc_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.layer_norm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            if self.dropout > 0:
                fc_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        self.latent_layer = nn.Linear(prev_dim, self.latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, channels, height, width)
        conv_features = self.conv_layers(x)
        flattened = conv_features.view(conv_features.size(0), -1)
        
        fc_features = self.fc_layers(flattened)
        fc_features = self._get_activation()(fc_features)
        
        latent = self.latent_layer(fc_features)
        return latent


class RNNEncoder(BaseEncoder):
    """Recurrent Neural Network Encoder for sequential inputs"""
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 rnn_type: str = 'lstm',
                 bidirectional: bool = True,
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
        
        super().__init__(input_dim=input_dim, latent_dim=latent_dim,
                        hidden_dims=fc_dims, activation=activation,
                        dropout=dropout, layer_norm=layer_norm, device=device)
    
    def _build_encoder(self):
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
        self.latent_layer = nn.Linear(prev_dim, self.latent_dim)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        # x shape: (batch_size, seq_len, input_dim)
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Use the last output
        last_output = rnn_out[:, -1, :]
        
        fc_features = self.fc_layers(last_output)
        fc_features = self._get_activation()(fc_features)
        
        latent = self.latent_layer(fc_features)
        return latent, hidden


class TransformerEncoder(BaseEncoder):
    """Transformer-based Encoder for complex sequential patterns"""
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
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
        
        super().__init__(input_dim=input_dim, latent_dim=latent_dim,
                        hidden_dims=fc_dims, activation=activation,
                        dropout=dropout, layer_norm=layer_norm, device=device)
    
    def _build_encoder(self):
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
        self.latent_layer = nn.Linear(prev_dim, self.latent_dim)
    
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
        
        latent = self.latent_layer(fc_features)
        return latent


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class VariationalEncoder(BaseEncoder):
    """Variational Autoencoder Encoder with reparameterization trick"""
    
    def _build_encoder(self):
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
        
        # Mean and log variance for reparameterization
        self.mu_layer = nn.Linear(prev_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, self.latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_layers(x)
        features = self._get_activation()(features)
        
        mu = self.mu_layer(features)
        logvar = self.logvar_layer(features)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar


class EncoderFactory:
    """Factory class for creating encoders"""
    
    @staticmethod
    def create_encoder(encoder_type: str, **kwargs) -> BaseEncoder:
        """Create an encoder based on type"""
        encoder_types = {
            'mlp': MLPEncoder,
            'cnn': CNNEncoder,
            'rnn': RNNEncoder,
            'transformer': TransformerEncoder,
            'vae': VariationalEncoder
        }
        
        if encoder_type not in encoder_types:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        return encoder_types[encoder_type](**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test MLP Encoder
    mlp_encoder = EncoderFactory.create_encoder(
        encoder_type='mlp',
        input_dim=100,
        latent_dim=32,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Test CNN Encoder
    cnn_encoder = EncoderFactory.create_encoder(
        encoder_type='cnn',
        input_channels=3,
        latent_dim=32,
        conv_dims=[32, 64, 128, 256],
        fc_dims=[512, 256]
    )
    
    # Test RNN Encoder
    rnn_encoder = EncoderFactory.create_encoder(
        encoder_type='rnn',
        input_dim=10,
        latent_dim=32,
        hidden_dim=256,
        num_layers=2,
        rnn_type='lstm',
        bidirectional=True
    )
    
    # Test VAE Encoder
    vae_encoder = EncoderFactory.create_encoder(
        encoder_type='vae',
        input_dim=100,
        latent_dim=32,
        hidden_dims=[256, 128],
        activation='relu',
        dropout=0.1
    )
    
    print("All encoders created successfully!") 