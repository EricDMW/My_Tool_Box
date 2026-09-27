"""Private mixins implementing the convolutional, recurrent and transformer trunks.

The policy, value and Q networks and the encoders share identical feature
extractors; only their output heads differ. The mixins below build the trunk
modules on ``self`` (keeping the public attribute names ``conv_layers``,
``rnn``, ``transformer``, ``fc_layers``, ...) and implement the corresponding
forward logic once.

A class using a mixin must set the documented configuration attributes before
``_build_*_trunk`` is called (the concrete classes do this in ``__init__``
before delegating to their base class).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import torch
import torch.nn as nn

from .layers import (
    ActivationSpec,
    PositionalEncoding,
    RNNHidden,
    _as_padding_mask,
    _check_dims,
    _check_dropout,
    _check_positive_int,
    _mlp_width,
    _per_layer,
    build_conv_stack,
    build_mlp,
    build_rnn,
    build_transformer_encoder,
    final_hidden_state,
    get_activation,
    masked_mean,
)

Device = Optional[Union[str, torch.device]]


def _init_common(
    module: nn.Module,
    hidden_dims: Sequence[int] | None,
    activation: ActivationSpec,
    dropout: float,
    layer_norm: bool,
    device: Device,
) -> None:
    """Validate and store the hyperparameters shared by every network class."""
    module.hidden_dims = _check_dims("hidden_dims", hidden_dims)
    get_activation(activation)  # raises ValueError for unknown names
    module.activation = activation
    module.dropout = _check_dropout(dropout)
    module.layer_norm = bool(layer_norm)
    module.device = device


def _finalize(module: nn.Module) -> None:
    """Move a freshly built module to its configured device."""
    if module.device is not None:
        module.to(module.device)


def _as_sequence(x: torch.Tensor, name: str) -> torch.Tensor:
    """Accept ``(batch, seq, features)`` or a single step ``(batch, features)``."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() != 3:
        raise ValueError(
            f"{name} expects input of shape (batch, seq_len, features) or (batch, features), "
            f"got {tuple(x.shape)}"
        )
    return x


class _ConvTrunk:
    """Conv stack -> adaptive average pooling -> flatten -> MLP.

    Required attributes: ``input_channels``, ``conv_dims``, ``kernel_sizes``,
    ``strides``, ``fc_dims``, ``activation``, ``dropout``, ``layer_norm``.
    ``layer_norm=True`` inserts ``BatchNorm2d`` in the convolutional blocks and
    ``LayerNorm`` in the fully connected blocks.
    """

    def _init_conv_config(
        self,
        input_channels: int,
        conv_dims: Sequence[int],
        kernel_sizes: int | Sequence[int] | None,
        strides: int | Sequence[int] | None,
        default_stride: int = 1,
        first_stride: int | None = None,
        pooled_size: int = 1,
    ) -> None:
        self.input_channels = _check_positive_int("input_channels", input_channels)
        self.conv_dims = _check_dims("conv_dims", conv_dims, allow_empty=False)
        n = len(self.conv_dims)
        default_strides = [default_stride] * n
        if first_stride is not None:
            default_strides[0] = first_stride
        self.kernel_sizes = _per_layer("kernel_sizes", kernel_sizes, n, [3] * n)
        self.strides = _per_layer("strides", strides, n, default_strides)
        self.pooled_size = _check_positive_int("pooled_size", pooled_size)

    def _build_conv_trunk(self) -> int:
        self.conv_layers = build_conv_stack(
            self.input_channels,
            self.conv_dims,
            self.kernel_sizes,
            self.strides,
            activation=self.activation,
            dropout=self.dropout,
            batch_norm=self.layer_norm,
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.pooled_size, self.pooled_size))
        flat = self.conv_dims[-1] * self.pooled_size * self.pooled_size
        self.fc_layers = build_mlp(
            flat,
            self.fc_dims,
            activation=self.activation,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )
        return _mlp_width(flat, self.fc_dims)

    def _conv_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"{type(self).__name__} expects images of shape (batch, channels, height, width), "
                f"got {tuple(x.shape)}"
            )
        h = self.adaptive_pool(self.conv_layers(x))
        return self.fc_layers(torch.flatten(h, 1))


class _RecurrentTrunk:
    """LSTM/GRU -> final hidden state of the last layer -> MLP.

    Required attributes: ``hidden_dim``, ``num_layers``, ``rnn_type``,
    ``bidirectional``, ``fc_dims``, ``activation``, ``dropout``, ``layer_norm``.
    """

    def _init_rnn_config(
        self, hidden_dim: int, num_layers: int, rnn_type: str, bidirectional: bool
    ) -> None:
        self.hidden_dim = _check_positive_int("hidden_dim", hidden_dim)
        self.num_layers = _check_positive_int("num_layers", num_layers)
        self.rnn_type = rnn_type
        self.bidirectional = bool(bidirectional)

    def _build_recurrent_trunk(self, input_size: int) -> int:
        self.rnn = build_rnn(
            self.rnn_type,
            input_size,
            self.hidden_dim,
            self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
        )
        rnn_out = self.hidden_dim * (2 if self.bidirectional else 1)
        self.fc_layers = build_mlp(
            rnn_out,
            self.fc_dims,
            activation=self.activation,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )
        return _mlp_width(rnn_out, self.fc_dims)

    def _recurrent_features(
        self, x: torch.Tensor, hidden: RNNHidden | None
    ) -> tuple[torch.Tensor, RNNHidden]:
        x = _as_sequence(x, type(self).__name__)
        _, hidden = self.rnn(x, hidden)
        return self.fc_layers(final_hidden_state(hidden, self.bidirectional)), hidden


class _TransformerTrunk:
    """Linear projection -> positional encoding -> transformer encoder -> masked mean -> MLP.

    Required attributes: ``d_model``, ``nhead``, ``num_layers``,
    ``dim_feedforward``, ``max_len``, ``fc_dims``, ``activation``, ``dropout``,
    ``layer_norm``.
    """

    def _init_transformer_config(
        self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, max_len: int
    ) -> None:
        self.d_model = _check_positive_int("d_model", d_model)
        self.nhead = _check_positive_int("nhead", nhead)
        if self.d_model % self.nhead:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.num_layers = _check_positive_int("num_layers", num_layers)
        self.dim_feedforward = _check_positive_int("dim_feedforward", dim_feedforward)
        self.max_len = _check_positive_int("max_len", max_len)

    def _build_transformer_trunk(self, input_size: int) -> int:
        self.input_projection = nn.Linear(
            _check_positive_int("input_dim", input_size), self.d_model
        )
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout, max_len=self.max_len)
        self.transformer = build_transformer_encoder(
            self.d_model, self.nhead, self.num_layers, self.dim_feedforward, self.dropout
        )
        self.fc_layers = build_mlp(
            self.d_model,
            self.fc_dims,
            activation=self.activation,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )
        return _mlp_width(self.d_model, self.fc_dims)

    def _transformer_features(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = _as_sequence(x, type(self).__name__)
        padding_mask = _as_padding_mask(mask, x)
        h = self.pos_encoder(self.input_projection(x))
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        return self.fc_layers(masked_mean(h, padding_mask))
