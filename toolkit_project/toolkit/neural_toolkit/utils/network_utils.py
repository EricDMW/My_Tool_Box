"""
Neural Network Utilities

This module provides utility functions for neural network operations, initialization,
regularization, and common patterns used in reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import math


class NetworkUtils:
    """Utility class for neural network operations"""
    
    @staticmethod
    def initialize_weights(module: nn.Module, method: str = 'xavier_uniform'):
        """Initialize network weights using specified method"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            if method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            elif method == 'orthogonal':
                nn.init.orthogonal_(module.weight, gain=1.0)
            elif method == 'uniform':
                nn.init.uniform_(module.weight, -0.1, 0.1)
            elif method == 'normal':
                nn.init.normal_(module.weight, 0.0, 0.1)
            else:
                raise ValueError(f"Unsupported initialization method: {method}")
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count the number of trainable parameters in a model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
        """Count parameters by layer name"""
        param_counts = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                param_counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return param_counts
    
    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: List[str]):
        """Freeze specific layers by name"""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    @staticmethod
    def unfreeze_layers(model: nn.Module, layer_names: List[str]):
        """Unfreeze specific layers by name"""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
    
    @staticmethod
    def get_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
        """Calculate gradient norm of model parameters"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm
    
    @staticmethod
    def clip_grad_norm(model: nn.Module, max_norm: float, norm_type: float = 2.0):
        """Clip gradient norm of model parameters"""
        nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)
    
    @staticmethod
    def create_mlp_layers(input_dim: int, hidden_dims: List[int], output_dim: int,
                         activation: str = 'relu', dropout: float = 0.0,
                         layer_norm: bool = False, bias: bool = True) -> nn.Sequential:
        """Create MLP layers with specified configuration"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(NetworkUtils.get_activation(activation))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        return nn.Sequential(*layers)
    
    @staticmethod
    def get_activation(activation: str) -> nn.Module:
        """Get activation function module"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),  # SiLU is the same as Swish
            'mish': nn.Mish(),
            'softplus': nn.Softplus(),
            'identity': nn.Identity()
        }
        return activations.get(activation, nn.ReLU())
    
    @staticmethod
    def create_conv_layers(input_channels: int, conv_dims: List[int],
                          kernel_sizes: List[int] = None, strides: List[int] = None,
                          padding: List[int] = None, activation: str = 'relu',
                          dropout: float = 0.0, layer_norm: bool = False) -> nn.Sequential:
        """Create convolutional layers with specified configuration"""
        if kernel_sizes is None:
            kernel_sizes = [3] * len(conv_dims)
        if strides is None:
            strides = [1] * len(conv_dims)
        if padding is None:
            padding = [k // 2 for k in kernel_sizes]
        
        layers = []
        prev_channels = input_channels
        
        for out_channels, kernel_size, stride, pad in zip(conv_dims, kernel_sizes, strides, padding):
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size, stride, pad),
                nn.BatchNorm2d(out_channels) if layer_norm else nn.Identity(),
                NetworkUtils.get_activation(activation),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            ])
            prev_channels = out_channels
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def create_transposed_conv_layers(input_channels: int, conv_dims: List[int],
                                    kernel_sizes: List[int] = None, strides: List[int] = None,
                                    padding: List[int] = None, output_padding: List[int] = None,
                                    activation: str = 'relu', dropout: float = 0.0,
                                    layer_norm: bool = False) -> nn.Sequential:
        """Create transposed convolutional layers with specified configuration"""
        if kernel_sizes is None:
            kernel_sizes = [3] * len(conv_dims)
        if strides is None:
            strides = [2] * len(conv_dims)
        if padding is None:
            padding = [k // 2 for k in kernel_sizes]
        if output_padding is None:
            output_padding = [s - 1 for s in strides]
        
        layers = []
        prev_channels = input_channels
        
        for out_channels, kernel_size, stride, pad, out_pad in zip(
            conv_dims, kernel_sizes, strides, padding, output_padding):
            layers.extend([
                nn.ConvTranspose2d(prev_channels, out_channels, kernel_size, stride, pad, out_pad),
                nn.BatchNorm2d(out_channels) if layer_norm else nn.Identity(),
                NetworkUtils.get_activation(activation),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            ])
            prev_channels = out_channels
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def create_attention_layer(input_dim: int, num_heads: int = 8, dropout: float = 0.1) -> nn.Module:
        """Create multi-head attention layer"""
        return nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
    
    @staticmethod
    def create_positional_encoding(d_model: int, max_len: int = 5000) -> nn.Module:
        """Create positional encoding for transformer models"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)
    
    @staticmethod
    def apply_weight_decay(model: nn.Module, weight_decay: float, skip_list: List[str] = None):
        """Apply weight decay to model parameters, excluding specified layers"""
        if skip_list is None:
            skip_list = []
        
        decay = []
        no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if len(param.shape) == 1 or any(skip_name in name for skip_name in skip_list):
                no_decay.append(param)
            else:
                decay.append(param)
        
        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
    
    @staticmethod
    def create_optimizer(model: nn.Module, optimizer_type: str = 'adam', 
                        lr: float = 1e-3, weight_decay: float = 0.0,
                        skip_list: List[str] = None) -> torch.optim.Optimizer:
        """Create optimizer with weight decay support"""
        if weight_decay > 0:
            param_groups = NetworkUtils.apply_weight_decay(model, weight_decay, skip_list)
        else:
            param_groups = model.parameters()
        
        optimizers = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
            'adagrad': torch.optim.Adagrad,
            'adamax': torch.optim.Adamax
        }
        
        optimizer_class = optimizers.get(optimizer_type.lower(), torch.optim.Adam)
        
        if optimizer_type.lower() == 'sgd':
            return optimizer_class(param_groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            return optimizer_class(param_groups, lr=lr, weight_decay=weight_decay)
    
    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str = 'step',
                        **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        schedulers = {
            'step': torch.optim.lr_scheduler.StepLR,
            'multistep': torch.optim.lr_scheduler.MultiStepLR,
            'exponential': torch.optim.lr_scheduler.ExponentialLR,
            'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
            'cosine_warm_restart': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'linear': torch.optim.lr_scheduler.LinearLR,
            'polynomial': torch.optim.lr_scheduler.PolynomialLR
        }
        
        scheduler_class = schedulers.get(scheduler_type.lower(), torch.optim.lr_scheduler.StepLR)
        return scheduler_class(optimizer, **kwargs)
    
    @staticmethod
    def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                       epoch: int, loss: float, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, filepath)
    
    @staticmethod
    def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                       filepath: str) -> Tuple[int, float]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss
    
    @staticmethod
    def compute_output_size(input_size: int, kernel_size: int, stride: int = 1, 
                           padding: int = 0, dilation: int = 1) -> int:
        """Compute output size for convolutional layer"""
        return ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    
    @staticmethod
    def compute_conv_output_size(input_size: Tuple[int, int], conv_layers: List[Dict]) -> Tuple[int, int]:
        """Compute output size after multiple convolutional layers"""
        h, w = input_size
        
        for layer_config in conv_layers:
            kernel_size = layer_config.get('kernel_size', 3)
            stride = layer_config.get('stride', 1)
            padding = layer_config.get('padding', kernel_size // 2)
            dilation = layer_config.get('dilation', 1)
            
            h = NetworkUtils.compute_output_size(h, kernel_size, stride, padding, dilation)
            w = NetworkUtils.compute_output_size(w, kernel_size, stride, padding, dilation)
        
        return h, w


# Example usage and testing
if __name__ == "__main__":
    # Test MLP creation
    mlp = NetworkUtils.create_mlp_layers(
        input_dim=10,
        hidden_dims=[256, 128],
        output_dim=4,
        activation='relu',
        dropout=0.1,
        layer_norm=True
    )
    
    # Test conv layers creation
    conv_layers = NetworkUtils.create_conv_layers(
        input_channels=3,
        conv_dims=[32, 64, 128],
        activation='relu',
        dropout=0.1
    )
    
    # Test optimizer creation
    optimizer = NetworkUtils.create_optimizer(
        mlp,
        optimizer_type='adam',
        lr=1e-3,
        weight_decay=1e-4
    )
    
    print("All network utilities created successfully!") 