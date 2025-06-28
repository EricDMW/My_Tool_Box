"""
Decoders for Neural Networks

This module provides flexible decoder implementations for reconstruction and generation tasks.
Supports various architectures including MLP, CNN, RNN, Transformer, and Autoencoder decoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod


class BaseDecoder(nn.Module, ABC):
    """Base class for all decoders"""
    
    def __init__(self, 
                 latent_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 device: str = 'cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.device = device
        
        self._build_decoder()
    
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
    def _build_decoder(self):
        """Build the decoder architecture"""
        pass
    
    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass


class MLPDecoder(BaseDecoder):
    """Multi-Layer Perceptron Decoder"""
    
    def _build_decoder(self):
        layers = []
        prev_dim = self.latent_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(z)
        features = self._get_activation()(features)
        output = self.output_layer(features)
        return output


class CNNDecoder(BaseDecoder):
    """Convolutional Neural Network Decoder for image generation"""
    
    def __init__(self,
                 latent_dim: int,
                 output_channels: int,
                 fc_dims: List[int] = [512, 256],
                 conv_dims: List[int] = [256, 128, 64, 32],
                 kernel_sizes: List[int] = [3, 3, 3, 3],
                 strides: List[int] = [2, 2, 2, 2],
                 initial_size: int = 4,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 device: str = 'cpu'):
        
        self.output_channels = output_channels
        self.fc_dims = fc_dims
        self.conv_dims = conv_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.initial_size = initial_size
        
        super().__init__(latent_dim=latent_dim, output_dim=0, 
                        hidden_dims=fc_dims, activation=activation,
                        dropout=dropout, layer_norm=layer_norm, device=device)
    
    def _build_decoder(self):
        # Build fully connected layers to expand latent space
        fc_layers = []
        prev_dim = self.latent_dim
        
        for hidden_dim in self.fc_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.layer_norm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            if self.dropout > 0:
                fc_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Final FC layer to get initial feature map
        self.initial_fc = nn.Linear(prev_dim, self.conv_dims[0] * self.initial_size * self.initial_size)
        
        # Build transposed convolutional layers
        conv_layers = []
        prev_channels = self.conv_dims[0]
        
        for i, (out_channels, kernel_size, stride) in enumerate(
            zip(self.conv_dims[1:], self.kernel_sizes[1:], self.strides[1:])):
            conv_layers.extend([
                nn.ConvTranspose2d(prev_channels, out_channels, kernel_size, stride, 
                                  padding=kernel_size//2, output_padding=stride-1),
                nn.BatchNorm2d(out_channels) if self.layer_norm else nn.Identity(),
                nn.ReLU() if self.activation == 'relu' else self._get_activation(),
                nn.Dropout2d(self.dropout) if self.dropout > 0 else nn.Identity()
            ])
            prev_channels = out_channels
        
        # Final output layer
        conv_layers.append(nn.ConvTranspose2d(prev_channels, self.output_channels, 
                                             self.kernel_sizes[-1], self.strides[-1],
                                             padding=self.kernel_sizes[-1]//2, 
                                             output_padding=self.strides[-1]-1))
        conv_layers.append(nn.Sigmoid())  # Output in [0, 1] range
        
        self.conv_layers = nn.Sequential(*conv_layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Expand latent space
        fc_features = self.fc_layers(z)
        fc_features = self._get_activation()(fc_features)
        
        # Reshape to initial feature map
        initial_features = self.initial_fc(fc_features)
        initial_features = initial_features.view(initial_features.size(0), 
                                               self.conv_dims[0], 
                                               self.initial_size, 
                                               self.initial_size)
        
        # Apply transposed convolutions
        output = self.conv_layers(initial_features)
        return output


class RNNDecoder(BaseDecoder):
    """Recurrent Neural Network Decoder for sequential generation"""
    
    def __init__(self,
                 latent_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 rnn_type: str = 'lstm',
                 max_seq_len: int = 100,
                 fc_dims: List[int] = [256],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 device: str = 'cpu'):
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.max_seq_len = max_seq_len
        self.fc_dims = fc_dims
        
        super().__init__(latent_dim=latent_dim, output_dim=output_dim,
                        hidden_dims=fc_dims, activation=activation,
                        dropout=dropout, layer_norm=layer_norm, device=device)
    
    def _build_decoder(self):
        # Initial projection from latent to RNN input
        self.latent_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        
        # Build RNN layer
        if self.rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                self.hidden_dim, self.hidden_dim, self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                self.hidden_dim, self.hidden_dim, self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
        
        # Build output layers
        fc_layers = []
        prev_dim = self.hidden_dim
        
        for hidden_dim in self.fc_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.layer_norm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            if self.dropout > 0:
                fc_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        batch_size = z.size(0)
        seq_len = seq_len or self.max_seq_len
        
        # Project latent to initial RNN input
        initial_input = self.latent_projection(z).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Expand to sequence length
        rnn_input = initial_input.expand(-1, seq_len, -1)
        
        # Apply RNN
        rnn_out, _ = self.rnn(rnn_input)
        
        # Apply output layers
        outputs = []
        for t in range(seq_len):
            features = self.fc_layers(rnn_out[:, t, :])
            features = self._get_activation()(features)
            output_t = self.output_layer(features)
            outputs.append(output_t)
        
        return torch.stack(outputs, dim=1)  # (batch, seq_len, output_dim)


class TransformerDecoder(BaseDecoder):
    """Transformer-based Decoder for complex sequential generation"""
    
    def __init__(self,
                 latent_dim: int,
                 output_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 max_seq_len: int = 100,
                 dropout: float = 0.1,
                 fc_dims: List[int] = [256],
                 activation: str = 'relu',
                 layer_norm: bool = True,
                 device: str = 'cpu'):
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len
        self.fc_dims = fc_dims
        
        super().__init__(latent_dim=latent_dim, output_dim=output_dim,
                        hidden_dims=fc_dims, activation=activation,
                        dropout=dropout, layer_norm=layer_norm, device=device)
    
    def _build_decoder(self):
        # Latent projection
        self.latent_projection = nn.Linear(self.latent_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, self.num_layers)
        
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
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        batch_size = z.size(0)
        seq_len = seq_len or self.max_seq_len
        
        # Project latent to transformer input
        latent_features = self.latent_projection(z).unsqueeze(1)  # (batch, 1, d_model)
        
        # Create target sequence (learned or positional)
        target_seq = torch.arange(seq_len, device=z.device).unsqueeze(0).expand(batch_size, -1)
        target_emb = self.pos_encoder(target_seq.float().unsqueeze(-1).expand(-1, -1, self.d_model))
        
        # Apply transformer decoder
        transformer_out = self.transformer(target_emb, latent_features)
        
        # Apply output layers
        outputs = []
        for t in range(seq_len):
            features = self.fc_layers(transformer_out[:, t, :])
            features = self._get_activation()(features)
            output_t = self.output_layer(features)
            outputs.append(output_t)
        
        return torch.stack(outputs, dim=1)  # (batch, seq_len, output_dim)


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
        pe_tensor = getattr(self, 'pe')
        x = x + pe_tensor[:x.size(0), :].to(x.device)
        return self.dropout(x)


class VariationalDecoder(BaseDecoder):
    """Variational Autoencoder Decoder with reconstruction loss"""
    
    def _build_decoder(self):
        layers = []
        prev_dim = self.latent_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Output layer (can be modified for different output types)
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(z)
        features = self._get_activation()(features)
        output = self.output_layer(features)
        return output


class DecoderFactory:
    """Factory class for creating decoders"""
    
    @staticmethod
    def create_decoder(decoder_type: str, **kwargs) -> BaseDecoder:
        """Create a decoder based on type"""
        decoder_types = {
            'mlp': MLPDecoder,
            'cnn': CNNDecoder,
            'rnn': RNNDecoder,
            'transformer': TransformerDecoder,
            'vae': VariationalDecoder
        }
        
        if decoder_type not in decoder_types:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        
        return decoder_types[decoder_type](**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test MLP Decoder
    mlp_decoder = DecoderFactory.create_decoder(
        decoder_type='mlp',
        latent_dim=32,
        output_dim=100,
        hidden_dims=[128, 256],
        activation='relu',
        dropout=0.1
    )
    
    # Test CNN Decoder
    cnn_decoder = DecoderFactory.create_decoder(
        decoder_type='cnn',
        latent_dim=32,
        output_channels=3,
        fc_dims=[512, 256],
        conv_dims=[256, 128, 64, 32],
        initial_size=4
    )
    
    # Test RNN Decoder
    rnn_decoder = DecoderFactory.create_decoder(
        decoder_type='rnn',
        latent_dim=32,
        output_dim=10,
        hidden_dim=256,
        num_layers=2,
        rnn_type='lstm',
        max_seq_len=50
    )
    
    # Test VAE Decoder
    vae_decoder = DecoderFactory.create_decoder(
        decoder_type='vae',
        latent_dim=32,
        output_dim=100,
        hidden_dims=[128, 256],
        activation='relu',
        dropout=0.1
    )
    
    print("All decoders created successfully!") 