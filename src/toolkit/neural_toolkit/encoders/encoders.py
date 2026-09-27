"""Encoders mapping observations to latent vectors.

=====================  ===============================  ===========================
Class                  Input shape                      Output
=====================  ===============================  ===========================
MLPEncoder             ``(batch, input_dim)``           ``(batch, latent_dim)``
CNNEncoder             ``(batch, C, H, W)``             ``(batch, latent_dim)``
RNNEncoder             ``(batch, seq_len, input_dim)``  ``((batch, latent_dim), hidden)``
TransformerEncoder     ``(batch, seq_len, input_dim)``  ``(batch, latent_dim)``
VariationalEncoder     ``(batch, input_dim)``           ``(z, mu, logvar)``, each ``(batch, latent_dim)``
=====================  ===============================  ===========================
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
    "BaseEncoder",
    "CNNEncoder",
    "EncoderFactory",
    "MLPEncoder",
    "PositionalEncoding",
    "RNNEncoder",
    "TransformerEncoder",
    "VariationalEncoder",
]


class BaseEncoder(nn.Module, ABC):
    """Base class for encoders.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    latent_dim : int
        Latent (code) dimension.
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
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = _check_positive_int("latent_dim", latent_dim)
        _init_common(self, hidden_dims, activation, dropout, layer_norm, device)
        self._build_encoder()
        _finalize(self)

    def _get_activation(self) -> nn.Module:
        """Return a new instance of the configured activation module."""
        return get_activation(self.activation)

    @abstractmethod
    def _build_encoder(self) -> None:
        """Create the sub-modules."""

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Forward pass; see the concrete subclasses for shapes."""


class MLPEncoder(BaseEncoder):
    """Multi-layer perceptron encoder (``feature_layers`` followed by ``latent_layer``)."""

    def _build_encoder(self) -> None:
        self.input_dim = _check_positive_int("input_dim", self.input_dim)
        self.feature_layers = build_mlp(
            self.input_dim,
            self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )
        self.latent_layer = nn.Linear(_mlp_width(self.input_dim, self.hidden_dims), self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent codes of shape ``(batch, latent_dim)``."""
        return self.latent_layer(self.feature_layers(x))


class CNNEncoder(_ConvTrunk, BaseEncoder):
    """Convolutional image encoder.

    Parameters
    ----------
    input_channels : int
        Image channels of inputs shaped ``(batch, C, H, W)``.
    latent_dim : int
        Latent dimension.
    conv_dims : sequence of int, default (32, 64, 128, 256)
        Output channels of the convolutional layers.
    fc_dims : sequence of int, default (512, 256)
        Hidden widths of the fully connected part.
    kernel_sizes : int or sequence of int, optional
        Per-layer kernel sizes (default 3).
    strides : int or sequence of int, optional
        Per-layer strides (default 1 for the first layer and 2 afterwards, i.e.
        ``(1, 2, 2, 2)`` for the default four layers).
    activation, dropout, layer_norm, device
        See :class:`BaseEncoder`.
    pooled_size : int, default 4
        The final feature map is adaptively average-pooled to
        ``pooled_size x pooled_size`` before flattening, so the encoder accepts
        any image size (for a 64x64 input and the default strides the map is 8x8).
    """

    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        conv_dims: Sequence[int] = (32, 64, 128, 256),
        fc_dims: Sequence[int] = (512, 256),
        kernel_sizes: int | Sequence[int] | None = None,
        strides: int | Sequence[int] | None = None,
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        device: Device = "cpu",
        pooled_size: int = 4,
    ) -> None:
        self._init_conv_config(
            input_channels,
            conv_dims,
            kernel_sizes,
            strides,
            default_stride=2,
            first_stride=1,
            pooled_size=pooled_size,
        )
        self.fc_dims = list(fc_dims)
        super().__init__(
            input_dim=0,
            latent_dim=latent_dim,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_encoder(self) -> None:
        self.latent_layer = nn.Linear(self._build_conv_trunk(), self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent codes of shape ``(batch, latent_dim)``."""
        return self.latent_layer(self._conv_features(x))


class RNNEncoder(_RecurrentTrunk, BaseEncoder):
    """Recurrent (LSTM/GRU) sequence encoder.

    Parameters are those of :class:`~toolkit.neural_toolkit.RNPolicyNetwork` with
    ``latent_dim`` in place of ``output_dim``; ``bidirectional`` defaults to True.
    ``forward(x, hidden=None)`` returns ``(latent, hidden)``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
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
            latent_dim=latent_dim,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_encoder(self) -> None:
        self.input_dim = _check_positive_int("input_dim", self.input_dim)
        self.latent_layer = nn.Linear(self._build_recurrent_trunk(self.input_dim), self.latent_dim)

    def forward(
        self, x: torch.Tensor, hidden: RNNHidden | None = None
    ) -> tuple[torch.Tensor, RNNHidden]:
        """Return ``(latent, hidden)`` with latent codes of shape ``(batch, latent_dim)``."""
        features, hidden = self._recurrent_features(x, hidden)
        return self.latent_layer(features), hidden


class TransformerEncoder(_TransformerTrunk, BaseEncoder):
    """Transformer sequence encoder (mean-pooled over valid positions).

    Parameters are those of :class:`~toolkit.neural_toolkit.TransformerPolicyNetwork`
    with ``latent_dim`` in place of ``output_dim``. ``forward(x, mask=None)`` accepts
    a boolean padding mask ``(batch, seq_len)`` (``True`` = padding).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
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
            latent_dim=latent_dim,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_encoder(self) -> None:
        self.input_dim = _check_positive_int("input_dim", self.input_dim)
        self.latent_layer = nn.Linear(
            self._build_transformer_trunk(self.input_dim), self.latent_dim
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return latent codes of shape ``(batch, latent_dim)``."""
        return self.latent_layer(self._transformer_features(x, mask))


class VariationalEncoder(BaseEncoder):
    """Gaussian VAE encoder with the reparameterisation trick.

    ``forward(x)`` returns ``(z, mu, logvar)`` where ``z = mu + exp(0.5 * logvar) * eps``
    with ``eps ~ N(0, I)`` drawn from PyTorch's default generator. Use
    :meth:`kl_divergence` for the KL term of the ELBO.
    """

    def _build_encoder(self) -> None:
        self.input_dim = _check_positive_int("input_dim", self.input_dim)
        self.feature_layers = build_mlp(
            self.input_dim,
            self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )
        width = _mlp_width(self.input_dim, self.hidden_dims)
        self.mu_layer = nn.Linear(width, self.latent_dim)
        self.logvar_layer = nn.Linear(width, self.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(z, mu, logvar)``, each of shape ``(batch, latent_dim)``."""
        features = self.feature_layers(x)
        mu = self.mu_layer(features)
        logvar = self.logvar_layer(features)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence ``KL(N(mu, exp(logvar)) || N(0, I))`` per sample, shape ``(batch,)``."""
        return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


class EncoderFactory:
    """Create encoders by name.

    Valid types: ``'mlp'``, ``'cnn'``, ``'rnn'``, ``'transformer'``, ``'vae'``.
    """

    ENCODER_TYPES: dict[str, type[BaseEncoder]] = {
        "mlp": MLPEncoder,
        "cnn": CNNEncoder,
        "rnn": RNNEncoder,
        "transformer": TransformerEncoder,
        "vae": VariationalEncoder,
    }

    @staticmethod
    def create_encoder(encoder_type: str, **kwargs) -> BaseEncoder:
        """Instantiate the encoder registered under ``encoder_type``.

        Raises
        ------
        ValueError
            If ``encoder_type`` is unknown.
        """
        key = str(encoder_type).lower()
        if key not in EncoderFactory.ENCODER_TYPES:
            valid = ", ".join(EncoderFactory.ENCODER_TYPES)
            raise ValueError(f"Unsupported encoder type {encoder_type!r}; valid types are: {valid}")
        return EncoderFactory.ENCODER_TYPES[key](**kwargs)
