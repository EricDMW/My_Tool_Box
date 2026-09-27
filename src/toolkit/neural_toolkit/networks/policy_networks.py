"""Policy networks for discrete and continuous action spaces.

All policies map observations to unnormalised outputs of size ``output_dim``:
logits for a categorical policy, or distribution parameters (for example means)
for a continuous policy. Interpreting the outputs is left to the algorithm.

=======================  ===============================  ===================
Class                    Input shape                      Output shape
=======================  ===============================  ===================
MLPPolicyNetwork         ``(batch, input_dim)``           ``(batch, output_dim)``
CNPolicyNetwork          ``(batch, C, H, W)``             ``(batch, output_dim)``
RNPolicyNetwork          ``(batch, seq_len, input_dim)``  ``((batch, output_dim), hidden)``
TransformerPolicyNetwork ``(batch, seq_len, input_dim)``  ``(batch, output_dim)``
=======================  ===============================  ===================

``CNNPolicyNetwork`` and ``RNNPolicyNetwork`` are aliases of the CNN and RNN classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn

from .._trunks import (
    Device,
    _ConvTrunk,
    _finalize,
    _init_common,
    _RecurrentTrunk,
    _TransformerTrunk,
)
from ..layers import (
    ActivationSpec,
    PositionalEncoding,
    RNNHidden,
    _check_positive_int,
    _mlp_width,
    build_mlp,
    get_activation,
)

__all__ = [
    "BasePolicyNetwork",
    "CNNPolicyNetwork",
    "CNPolicyNetwork",
    "MLPPolicyNetwork",
    "PolicyFactory",
    "PositionalEncoding",
    "RNNPolicyNetwork",
    "RNPolicyNetwork",
    "TransformerPolicyNetwork",
]


class BasePolicyNetwork(nn.Module, ABC):
    """Base class for policy networks.

    Parameters
    ----------
    input_dim : int
        Observation feature dimension.
    output_dim : int
        Number of policy outputs (actions, or distribution parameters).
    hidden_dims : sequence of int, default (256, 256)
        Hidden layer widths of the MLP part.
    activation : str or type, default ``'relu'``
        Hidden activation, see :func:`toolkit.neural_toolkit.layers.get_activation`.
    dropout : float, default 0.0
        Dropout probability in ``[0, 1)``.
    layer_norm : bool, default False
        Insert normalisation layers after each hidden linear layer.
    device : str or torch.device, default ``'cpu'``
        Device the module is moved to at the end of construction (``None`` leaves it).

    Raises
    ------
    ValueError, TypeError
        For invalid dimensions, dropout or activation names.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = _check_positive_int("output_dim", output_dim)
        _init_common(self, hidden_dims, activation, dropout, layer_norm, device)
        self._build_network()
        _finalize(self)

    def _get_activation(self) -> nn.Module:
        """Return a new instance of the configured activation module."""
        return get_activation(self.activation)

    @abstractmethod
    def _build_network(self) -> None:
        """Create the sub-modules."""

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Forward pass; see the concrete subclasses for shapes."""


class MLPPolicyNetwork(BasePolicyNetwork):
    """Multi-layer perceptron policy.

    ``feature_layers`` is the hidden stack and ``output_layer`` the final linear
    layer. See :class:`BasePolicyNetwork` for the parameters.

    Examples
    --------
    >>> net = MLPPolicyNetwork(input_dim=8, output_dim=4, hidden_dims=(64, 64))
    >>> net(torch.randn(32, 8)).shape
    torch.Size([32, 4])
    """

    def _build_network(self) -> None:
        self.input_dim = _check_positive_int("input_dim", self.input_dim)
        self.feature_layers = build_mlp(
            self.input_dim,
            self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )
        self.output_layer = nn.Linear(_mlp_width(self.input_dim, self.hidden_dims), self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return policy outputs of shape ``(batch, output_dim)``."""
        return self.output_layer(self.feature_layers(x))


class CNPolicyNetwork(_ConvTrunk, BasePolicyNetwork):
    """Convolutional policy for image observations.

    Parameters
    ----------
    input_channels : int
        Image channels ``C`` of inputs shaped ``(batch, C, H, W)``.
    output_dim : int
        Number of policy outputs.
    conv_dims : sequence of int, default (32, 64, 128)
        Output channels of the convolutional layers.
    fc_dims : sequence of int, default (256, 256)
        Hidden widths of the fully connected head.
    kernel_sizes, strides : int or sequence of int, optional
        Per-layer kernel sizes (default 3) and strides (default 1).
    activation, dropout, layer_norm, device
        See :class:`BasePolicyNetwork`. ``layer_norm`` adds ``BatchNorm2d`` to the
        convolutional blocks and ``LayerNorm`` to the fully connected blocks.

    Notes
    -----
    The feature map is globally average-pooled, so any image size works.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        conv_dims: Sequence[int] = (32, 64, 128),
        fc_dims: Sequence[int] = (256, 256),
        kernel_sizes: int | Sequence[int] | None = None,
        strides: int | Sequence[int] | None = None,
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        device: Device = "cpu",
    ) -> None:
        self._init_conv_config(input_channels, conv_dims, kernel_sizes, strides, default_stride=1)
        self.fc_dims = list(fc_dims)
        super().__init__(
            input_dim=0,
            output_dim=output_dim,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_network(self) -> None:
        self.output_layer = nn.Linear(self._build_conv_trunk(), self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return policy outputs of shape ``(batch, output_dim)`` for images ``(batch, C, H, W)``."""
        return self.output_layer(self._conv_features(x))


class RNPolicyNetwork(_RecurrentTrunk, BasePolicyNetwork):
    """Recurrent (LSTM/GRU) policy for sequential observations.

    Parameters
    ----------
    input_dim : int
        Observation feature dimension.
    output_dim : int
        Number of policy outputs.
    hidden_dim : int, default 256
        RNN hidden size.
    num_layers : int, default 2
        Stacked RNN layers.
    rnn_type : {'lstm', 'gru'}, default 'lstm'
        Recurrent cell type.
    bidirectional : bool, default False
        Use a bidirectional RNN.
    fc_dims : sequence of int, default (256,)
        Hidden widths of the fully connected head.
    activation, dropout, layer_norm, device
        See :class:`BasePolicyNetwork`.

    Notes
    -----
    The head reads the final hidden state of the last RNN layer (both directions
    when bidirectional). ``forward`` accepts ``(batch, seq_len, input_dim)`` or a
    single step ``(batch, input_dim)`` and returns ``(outputs, hidden)`` so the
    hidden state can be carried across calls.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        fc_dims: Sequence[int] = (256,),
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        device: Device = "cpu",
    ) -> None:
        self._init_rnn_config(hidden_dim, num_layers, rnn_type, bidirectional)
        self.fc_dims = list(fc_dims)
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_network(self) -> None:
        self.input_dim = _check_positive_int("input_dim", self.input_dim)
        self.output_layer = nn.Linear(self._build_recurrent_trunk(self.input_dim), self.output_dim)

    def forward(
        self, x: torch.Tensor, hidden: RNNHidden | None = None
    ) -> tuple[torch.Tensor, RNNHidden]:
        """Return ``(outputs, hidden)`` with outputs of shape ``(batch, output_dim)``."""
        features, hidden = self._recurrent_features(x, hidden)
        return self.output_layer(features), hidden


class TransformerPolicyNetwork(_TransformerTrunk, BasePolicyNetwork):
    """Transformer-encoder policy for sequential observations.

    Parameters
    ----------
    input_dim : int
        Observation feature dimension.
    output_dim : int
        Number of policy outputs.
    d_model : int, default 256
        Transformer width (must be divisible by ``nhead``).
    nhead : int, default 8
        Attention heads.
    num_layers : int, default 6
        Encoder layers.
    dim_feedforward : int, default 1024
        Feed-forward width inside each encoder layer.
    dropout : float, default 0.1
        Dropout in the encoder, positional encoding and head.
    fc_dims : sequence of int, default (256,)
        Hidden widths of the fully connected head.
    activation : str or type, default ``'relu'``
        Head activation.
    layer_norm : bool, default True
        ``LayerNorm`` in the head.
    device : str or torch.device, default ``'cpu'``
        Target device.
    max_len : int, default 5000
        Longest supported sequence (size of the positional-encoding table).

    Notes
    -----
    ``forward(x, mask=None)`` takes ``x`` of shape ``(batch, seq_len, input_dim)``
    (or ``(batch, input_dim)`` for a single step) and an optional boolean padding
    mask ``(batch, seq_len)`` where ``True`` marks padded positions. Encoder
    outputs are mean-pooled over valid positions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        fc_dims: Sequence[int] = (256,),
        activation: ActivationSpec = "relu",
        layer_norm: bool = True,
        device: Device = "cpu",
        max_len: int = 5000,
    ) -> None:
        self._init_transformer_config(d_model, nhead, num_layers, dim_feedforward, max_len)
        self.fc_dims = list(fc_dims)
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_network(self) -> None:
        self.input_dim = _check_positive_int("input_dim", self.input_dim)
        self.output_layer = nn.Linear(
            self._build_transformer_trunk(self.input_dim), self.output_dim
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return policy outputs of shape ``(batch, output_dim)``."""
        return self.output_layer(self._transformer_features(x, mask))


CNNPolicyNetwork = CNPolicyNetwork
RNNPolicyNetwork = RNPolicyNetwork


class PolicyFactory:
    """Create policy networks by name.

    Valid types: ``'mlp'``, ``'cnn'``, ``'rnn'``, ``'transformer'``.
    """

    POLICY_TYPES: dict[str, type[BasePolicyNetwork]] = {
        "mlp": MLPPolicyNetwork,
        "cnn": CNPolicyNetwork,
        "rnn": RNPolicyNetwork,
        "transformer": TransformerPolicyNetwork,
    }

    @staticmethod
    def create_policy(policy_type: str, **kwargs) -> BasePolicyNetwork:
        """Instantiate the policy class registered under ``policy_type``.

        Parameters
        ----------
        policy_type : str
            One of ``'mlp'``, ``'cnn'``, ``'rnn'``, ``'transformer'`` (case-insensitive).
        **kwargs
            Constructor arguments of the selected class.

        Raises
        ------
        ValueError
            If ``policy_type`` is unknown.

        Examples
        --------
        >>> PolicyFactory.create_policy("mlp", input_dim=4, output_dim=2).output_dim
        2
        """
        key = str(policy_type).lower()
        if key not in PolicyFactory.POLICY_TYPES:
            valid = ", ".join(PolicyFactory.POLICY_TYPES)
            raise ValueError(f"Unsupported policy type {policy_type!r}; valid types are: {valid}")
        return PolicyFactory.POLICY_TYPES[key](**kwargs)
