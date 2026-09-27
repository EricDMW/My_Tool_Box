"""Decoders mapping latent vectors back to observations or sequences.

=====================  ======================  =====================================
Class                  Input shape             Output shape
=====================  ======================  =====================================
MLPDecoder             ``(batch, latent_dim)``  ``(batch, output_dim)``
CNNDecoder             ``(batch, latent_dim)``  ``(batch, output_channels, S, S)``
RNNDecoder             ``(batch, latent_dim)``  ``(batch, seq_len, output_dim)``
TransformerDecoder     ``(batch, latent_dim)``  ``(batch, seq_len, output_dim)``
VariationalDecoder     ``(batch, latent_dim)``  ``(batch, output_dim)``
=====================  ======================  =====================================

For :class:`CNNDecoder`, ``S = initial_size * prod(strides)`` (64 for the defaults).
The sequence decoders produce ``max_seq_len`` steps unless ``seq_len`` is passed.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn

from .._trunks import Device, _finalize, _init_common, _RecurrentTrunk
from ..layers import (
    ActivationSpec,
    PositionalEncoding,
    _check_dims,
    _check_positive_int,
    _mlp_width,
    _per_layer,
    _transposed_padding,
    build_mlp,
    get_activation,
)

__all__ = [
    "BaseDecoder",
    "CNNDecoder",
    "DecoderFactory",
    "MLPDecoder",
    "PositionalEncoding",
    "RNNDecoder",
    "TransformerDecoder",
    "VariationalDecoder",
]


def _check_seq_len(seq_len: int | None, default: int) -> int:
    return default if seq_len is None else _check_positive_int("seq_len", seq_len)


class BaseDecoder(nn.Module, ABC):
    """Base class for decoders.

    Parameters
    ----------
    latent_dim : int
        Latent (code) dimension.
    output_dim : int
        Output feature dimension.
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
        latent_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.latent_dim = _check_positive_int("latent_dim", latent_dim)
        self.output_dim = output_dim
        _init_common(self, hidden_dims, activation, dropout, layer_norm, device)
        self._build_decoder()
        _finalize(self)

    def _get_activation(self) -> nn.Module:
        """Return a new instance of the configured activation module."""
        return get_activation(self.activation)

    @abstractmethod
    def _build_decoder(self) -> None:
        """Create the sub-modules."""

    @abstractmethod
    def forward(self, z: torch.Tensor):
        """Forward pass; see the concrete subclasses for shapes."""


class MLPDecoder(BaseDecoder):
    """Multi-layer perceptron decoder (``feature_layers`` followed by ``output_layer``)."""

    def _build_decoder(self) -> None:
        self.output_dim = _check_positive_int("output_dim", self.output_dim)
        self.feature_layers = build_mlp(
            self.latent_dim,
            self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )
        self.output_layer = nn.Linear(
            _mlp_width(self.latent_dim, self.hidden_dims), self.output_dim
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return reconstructions of shape ``(batch, output_dim)``."""
        return self.output_layer(self.feature_layers(z))


class CNNDecoder(BaseDecoder):
    """Transposed-convolution image decoder.

    The latent vector passes through an MLP (``fc_layers``), is projected by
    ``initial_fc`` (followed by the activation) to a ``conv_dims[0] x initial_size x
    initial_size`` feature map and upsampled by one ``ConvTranspose2d`` per entry of
    ``conv_dims``: ``conv_dims[i] -> conv_dims[i + 1]`` and finally
    ``conv_dims[-1] -> output_channels``. Each layer scales the spatial size by its
    stride exactly, so the output is ``(batch, output_channels, S, S)`` with
    ``S = initial_size * prod(strides)`` (see :attr:`output_size`).

    Parameters
    ----------
    latent_dim : int
        Latent dimension.
    output_channels : int
        Channels of the generated image.
    fc_dims : sequence of int, default (512, 256)
        Hidden widths of the fully connected part.
    conv_dims : sequence of int, default (256, 128, 64, 32)
        Channel counts; ``conv_dims[0]`` is the channel count of the initial map.
    kernel_sizes : int or sequence of int, optional
        Per-layer kernel sizes (default 3), one per entry of ``conv_dims``.
    strides : int or sequence of int, optional
        Per-layer strides (default 2), one per entry of ``conv_dims``.
    initial_size : int, default 4
        Spatial size of the initial feature map.
    activation, dropout, layer_norm, device
        See :class:`BaseDecoder`. ``layer_norm`` adds ``BatchNorm2d`` to the
        convolutional blocks and ``LayerNorm`` to the fully connected blocks.
    output_activation : str or type, optional, default ``'sigmoid'``
        Activation applied to the output image (``None`` keeps it linear).
    """

    def __init__(
        self,
        latent_dim: int,
        output_channels: int,
        fc_dims: Sequence[int] = (512, 256),
        conv_dims: Sequence[int] = (256, 128, 64, 32),
        kernel_sizes: int | Sequence[int] | None = None,
        strides: int | Sequence[int] | None = None,
        initial_size: int = 4,
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        device: Device = "cpu",
        output_activation: ActivationSpec | None = "sigmoid",
    ) -> None:
        self.output_channels = _check_positive_int("output_channels", output_channels)
        self.fc_dims = list(fc_dims)
        self.conv_dims = _check_dims("conv_dims", conv_dims, allow_empty=False)
        n = len(self.conv_dims)
        self.kernel_sizes = _per_layer("kernel_sizes", kernel_sizes, n, [3] * n)
        self.strides = _per_layer("strides", strides, n, [2] * n)
        self.initial_size = _check_positive_int("initial_size", initial_size)
        if output_activation is not None:
            get_activation(output_activation)
        self.output_activation = output_activation
        super().__init__(
            latent_dim=latent_dim,
            output_dim=0,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    @property
    def output_size(self) -> int:
        """Height and width of the generated images."""
        return self.initial_size * math.prod(self.strides)

    def _build_decoder(self) -> None:
        self.fc_layers = build_mlp(
            self.latent_dim,
            self.fc_dims,
            activation=self.activation,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )
        self.initial_fc = nn.Linear(
            _mlp_width(self.latent_dim, self.fc_dims),
            self.conv_dims[0] * self.initial_size * self.initial_size,
        )
        self.initial_activation = get_activation(self.activation)

        layers: list[nn.Module] = []
        channels = [*self.conv_dims, self.output_channels]
        n = len(self.conv_dims)
        for i, (kernel, stride) in enumerate(zip(self.kernel_sizes, self.strides)):
            padding, output_padding = _transposed_padding(kernel, stride)
            layers.append(
                nn.ConvTranspose2d(
                    channels[i],
                    channels[i + 1],
                    kernel,
                    stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            if i < n - 1:
                if self.layer_norm:
                    layers.append(nn.BatchNorm2d(channels[i + 1]))
                layers.append(get_activation(self.activation))
                if self.dropout > 0.0:
                    layers.append(nn.Dropout2d(self.dropout))
        if self.output_activation is not None:
            layers.append(get_activation(self.output_activation))
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return images of shape ``(batch, output_channels, output_size, output_size)``."""
        h = self.initial_activation(self.initial_fc(self.fc_layers(z)))
        h = h.view(z.size(0), self.conv_dims[0], self.initial_size, self.initial_size)
        return self.conv_layers(h)


class RNNDecoder(_RecurrentTrunk, BaseDecoder):
    """Recurrent sequence decoder.

    The projected latent vector is fed to a unidirectional LSTM/GRU at every time
    step; a shared MLP head maps each RNN output to ``output_dim`` features.

    Parameters
    ----------
    latent_dim : int
        Latent dimension.
    output_dim : int
        Features per generated time step.
    hidden_dim : int, default 256
        RNN hidden size.
    num_layers : int, default 2
        Stacked RNN layers.
    rnn_type : {'lstm', 'gru'}, default 'lstm'
        Recurrent cell type.
    max_seq_len : int, default 100
        Number of steps generated when ``forward`` is called without ``seq_len``.
    fc_dims : sequence of int, default (256,)
        Hidden widths of the output head.
    activation, dropout, layer_norm, device
        See :class:`BaseDecoder`.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        max_seq_len: int = 100,
        fc_dims: Sequence[int] = (256,),
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        device: Device = "cpu",
    ) -> None:
        self._init_rnn_config(hidden_dim, num_layers, rnn_type, bidirectional=False)
        self.max_seq_len = _check_positive_int("max_seq_len", max_seq_len)
        self.fc_dims = list(fc_dims)
        super().__init__(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_decoder(self) -> None:
        self.output_dim = _check_positive_int("output_dim", self.output_dim)
        self.latent_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self._build_recurrent_trunk(self.hidden_dim), self.output_dim)

    def forward(self, z: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        """Return sequences of shape ``(batch, seq_len, output_dim)``."""
        steps = _check_seq_len(seq_len, self.max_seq_len)
        rnn_input = self.latent_projection(z).unsqueeze(1).expand(-1, steps, -1)
        rnn_out, _ = self.rnn(rnn_input)
        return self.output_layer(self.fc_layers(rnn_out))


class TransformerDecoder(BaseDecoder):
    """Non-autoregressive transformer sequence decoder.

    The queries are sinusoidal positional encodings of the output positions; the
    memory is the projected latent vector (a single token). Every position is
    decoded in parallel by ``nn.TransformerDecoder`` and mapped to ``output_dim``
    features by a shared MLP head.

    Parameters
    ----------
    latent_dim : int
        Latent dimension.
    output_dim : int
        Features per generated time step.
    d_model, nhead, num_layers, dim_feedforward, dropout
        Transformer hyperparameters (``d_model`` divisible by ``nhead``).
    max_seq_len : int, default 100
        Number of steps generated when ``forward`` is called without ``seq_len``.
    fc_dims : sequence of int, default (256,)
        Hidden widths of the output head.
    activation : str or type, default ``'relu'``
        Head activation.
    layer_norm : bool, default True
        ``LayerNorm`` in the head.
    device : str or torch.device, default ``'cpu'``
        Target device.
    max_len : int, optional
        Size of the positional-encoding table (default ``max(max_seq_len, 5000)``).
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_seq_len: int = 100,
        dropout: float = 0.1,
        fc_dims: Sequence[int] = (256,),
        activation: ActivationSpec = "relu",
        layer_norm: bool = True,
        device: Device = "cpu",
        max_len: int | None = None,
    ) -> None:
        self.d_model = _check_positive_int("d_model", d_model)
        self.nhead = _check_positive_int("nhead", nhead)
        if self.d_model % self.nhead:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.num_layers = _check_positive_int("num_layers", num_layers)
        self.dim_feedforward = _check_positive_int("dim_feedforward", dim_feedforward)
        self.max_seq_len = _check_positive_int("max_seq_len", max_seq_len)
        self.max_len = (
            max(self.max_seq_len, 5000)
            if max_len is None
            else _check_positive_int("max_len", max_len)
        )
        self.fc_dims = list(fc_dims)
        super().__init__(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_decoder(self) -> None:
        self.output_dim = _check_positive_int("output_dim", self.output_dim)
        self.latent_projection = nn.Linear(self.latent_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout, max_len=self.max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, self.num_layers)
        self.fc_layers = build_mlp(
            self.d_model,
            self.fc_dims,
            activation=self.activation,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )
        self.output_layer = nn.Linear(_mlp_width(self.d_model, self.fc_dims), self.output_dim)

    def forward(self, z: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        """Return sequences of shape ``(batch, seq_len, output_dim)``."""
        steps = _check_seq_len(seq_len, self.max_seq_len)
        memory = self.latent_projection(z).unsqueeze(1)  # (batch, 1, d_model)
        queries = self.pos_encoder(memory.new_zeros(z.size(0), steps, self.d_model))
        decoded = self.transformer(queries, memory)
        return self.output_layer(self.fc_layers(decoded))


class VariationalDecoder(MLPDecoder):
    """Gaussian VAE decoder: an MLP from ``z`` to the reconstruction ``(batch, output_dim)``."""


class DecoderFactory:
    """Create decoders by name.

    Valid types: ``'mlp'``, ``'cnn'``, ``'rnn'``, ``'transformer'``, ``'vae'``.
    """

    DECODER_TYPES: dict[str, type[BaseDecoder]] = {
        "mlp": MLPDecoder,
        "cnn": CNNDecoder,
        "rnn": RNNDecoder,
        "transformer": TransformerDecoder,
        "vae": VariationalDecoder,
    }

    @staticmethod
    def create_decoder(decoder_type: str, **kwargs) -> BaseDecoder:
        """Instantiate the decoder registered under ``decoder_type``.

        Raises
        ------
        ValueError
            If ``decoder_type`` is unknown.
        """
        key = str(decoder_type).lower()
        if key not in DecoderFactory.DECODER_TYPES:
            valid = ", ".join(DecoderFactory.DECODER_TYPES)
            raise ValueError(f"Unsupported decoder type {decoder_type!r}; valid types are: {valid}")
        return DecoderFactory.DECODER_TYPES[key](**kwargs)
