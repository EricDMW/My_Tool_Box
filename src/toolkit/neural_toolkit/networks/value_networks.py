"""State-value networks ``V(s)``.

Value networks map observations to ``output_dim`` value estimates (1 by default).

=======================  ===============================  ===================
Class                    Input shape                      Output shape
=======================  ===============================  ===================
MLPValueNetwork          ``(batch, input_dim)``           ``(batch, output_dim)``
CNNValueNetwork          ``(batch, C, H, W)``             ``(batch, output_dim)``
RNNValueNetwork          ``(batch, seq_len, input_dim)``  ``((batch, output_dim), hidden)``
TransformerValueNetwork  ``(batch, seq_len, input_dim)``  ``(batch, output_dim)``
=======================  ===============================  ===================
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
    "BaseValueNetwork",
    "CNNValueNetwork",
    "MLPValueNetwork",
    "PositionalEncoding",
    "RNNValueNetwork",
    "TransformerValueNetwork",
    "ValueFactory",
]


class BaseValueNetwork(nn.Module, ABC):
    """Base class for value networks.

    Parameters
    ----------
    input_dim : int
        Observation feature dimension.
    output_dim : int, default 1
        Number of value outputs (for example several reward heads).
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
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
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


class MLPValueNetwork(BaseValueNetwork):
    """Multi-layer perceptron value network.

    Examples
    --------
    >>> net = MLPValueNetwork(input_dim=8)
    >>> net(torch.randn(32, 8)).shape
    torch.Size([32, 1])
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
        """Return values of shape ``(batch, output_dim)``."""
        return self.output_layer(self.feature_layers(x))


class CNNValueNetwork(_ConvTrunk, BaseValueNetwork):
    """Convolutional value network for image observations.

    Parameters are those of :class:`~toolkit.neural_toolkit.CNPolicyNetwork`
    with ``output_dim`` defaulting to 1.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int = 1,
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
        """Return values of shape ``(batch, output_dim)`` for images ``(batch, C, H, W)``."""
        return self.output_layer(self._conv_features(x))


class RNNValueNetwork(_RecurrentTrunk, BaseValueNetwork):
    """Recurrent (LSTM/GRU) value network.

    Parameters are those of :class:`~toolkit.neural_toolkit.RNPolicyNetwork`
    with ``output_dim`` defaulting to 1. ``forward(x, hidden=None)`` returns
    ``(values, hidden)``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
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
        """Return ``(values, hidden)`` with values of shape ``(batch, output_dim)``."""
        features, hidden = self._recurrent_features(x, hidden)
        return self.output_layer(features), hidden


class TransformerValueNetwork(_TransformerTrunk, BaseValueNetwork):
    """Transformer-encoder value network.

    Parameters are those of :class:`~toolkit.neural_toolkit.TransformerPolicyNetwork`
    with ``output_dim`` defaulting to 1. ``forward(x, mask=None)`` returns values
    of shape ``(batch, output_dim)``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
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
        """Return values of shape ``(batch, output_dim)``."""
        return self.output_layer(self._transformer_features(x, mask))


class ValueFactory:
    """Create value networks by name.

    Valid types: ``'mlp'``, ``'cnn'``, ``'rnn'``, ``'transformer'``.
    """

    NETWORK_TYPES: dict[str, type[BaseValueNetwork]] = {
        "mlp": MLPValueNetwork,
        "cnn": CNNValueNetwork,
        "rnn": RNNValueNetwork,
        "transformer": TransformerValueNetwork,
    }

    @staticmethod
    def create_value_network(network_type: str, **kwargs) -> BaseValueNetwork:
        """Instantiate the value network registered under ``network_type``.

        Raises
        ------
        ValueError
            If ``network_type`` is unknown.
        """
        key = str(network_type).lower()
        if key not in ValueFactory.NETWORK_TYPES:
            valid = ", ".join(ValueFactory.NETWORK_TYPES)
            raise ValueError(f"Unsupported network type {network_type!r}; valid types are: {valid}")
        return ValueFactory.NETWORK_TYPES[key](**kwargs)
