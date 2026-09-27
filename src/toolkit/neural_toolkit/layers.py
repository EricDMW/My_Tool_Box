"""Shared building blocks for the neural toolkit.

Every network, encoder and decoder in :mod:`toolkit.neural_toolkit` is assembled
from the helpers in this module, so that activation lookup, MLP stacks,
positional encodings, recurrent and transformer trunks behave identically
everywhere.

The MLP convention used throughout the toolkit is::

    Linear -> [LayerNorm] -> activation -> [Dropout]    (per hidden layer)
    Linear                                              (output layer, no activation)

Examples
--------
>>> import torch
>>> from toolkit.neural_toolkit.layers import build_mlp, get_activation
>>> mlp = build_mlp(4, [32, 32], output_dim=2, activation="tanh")
>>> mlp(torch.zeros(8, 4)).shape
torch.Size([8, 2])
>>> get_activation("swish")
SiLU()
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Sequence
from typing import Callable, Union

import torch
import torch.nn as nn

__all__ = [
    "ACTIVATION_NAMES",
    "RNN_TYPES",
    "ActivationSpec",
    "PositionalEncoding",
    "build_conv_stack",
    "build_mlp",
    "build_rnn",
    "build_transformer_encoder",
    "final_hidden_state",
    "get_activation",
    "masked_mean",
    "sinusoidal_table",
]

ActivationSpec = Union[str, type[nn.Module]]
"""An activation name (see :data:`ACTIVATION_NAMES`) or an ``nn.Module`` subclass."""

RNNHidden = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "softplus": nn.Softplus,
    "identity": nn.Identity,
}

ACTIVATION_NAMES: tuple[str, ...] = tuple(sorted(_ACTIVATIONS))
"""Valid activation names accepted by :func:`get_activation`."""

RNN_TYPES: tuple[str, ...] = ("lstm", "gru")
"""Valid recurrent cell types accepted by :func:`build_rnn`."""


# ---------------------------------------------------------------------------
# Validation helpers (shared by all network modules)
# ---------------------------------------------------------------------------


def _check_positive_int(name: str, value: object) -> int:
    """Return ``value`` as ``int`` or raise if it is not a positive integer."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    try:
        ivalue = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{name} must be a positive integer, got {value!r}") from None
    if ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return ivalue


def _check_dims(name: str, dims: Iterable[int] | None, allow_empty: bool = True) -> list[int]:
    """Validate a sequence of layer widths and return it as a list of ints."""
    if dims is None:
        dims = ()
    if isinstance(dims, (int, str, bytes)):
        raise TypeError(f"{name} must be a sequence of positive integers, got {dims!r}")
    out = [_check_positive_int(f"{name}[{i}]", d) for i, d in enumerate(dims)]
    if not out and not allow_empty:
        raise ValueError(f"{name} must contain at least one entry")
    return out


def _check_dropout(value: float) -> float:
    """Validate a dropout probability in ``[0, 1)``."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"dropout must be a float in [0, 1), got {value!r}") from None
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {value!r}")
    return p


def _per_layer(
    name: str, value: int | Sequence[int] | None, n_layers: int, default: Sequence[int]
) -> list[int]:
    """Expand a per-layer hyperparameter (``None``, an int, or a sequence of length ``n_layers``)."""
    if value is None:
        values = list(default)
    elif isinstance(value, int) and not isinstance(value, bool):
        values = [value] * n_layers
    else:
        values = list(value)
    if len(values) != n_layers:
        raise ValueError(
            f"{name} must have one entry per convolutional layer ({n_layers}), "
            f"got {len(values)}: {values!r}"
        )
    return [_check_positive_int(f"{name}[{i}]", v) for i, v in enumerate(values)]


# ---------------------------------------------------------------------------
# Activations and MLPs
# ---------------------------------------------------------------------------


def get_activation(name: ActivationSpec) -> nn.Module:
    """Return a new activation module.

    Parameters
    ----------
    name : str or type
        Case-insensitive activation name, one of :data:`ACTIVATION_NAMES`
        (``'swish'`` and ``'silu'`` both map to :class:`torch.nn.SiLU`), or an
        ``nn.Module`` subclass that is instantiated without arguments.

    Returns
    -------
    torch.nn.Module
        A fresh module instance, safe to place inside ``nn.Sequential``.

    Raises
    ------
    ValueError
        If ``name`` is not a known activation name.
    TypeError
        If ``name`` is neither a string nor an ``nn.Module`` subclass.

    Examples
    --------
    >>> get_activation("gelu")
    GELU(approximate='none')
    """
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name()
    if not isinstance(name, str):
        raise TypeError(
            f"activation must be a string or an nn.Module subclass, got {type(name).__name__}"
        )
    key = name.strip().lower()
    try:
        factory = _ACTIVATIONS[key]
    except KeyError:
        raise ValueError(
            f"Unknown activation {name!r}; valid names are: {', '.join(ACTIVATION_NAMES)}"
        ) from None
    return factory()


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int | None = None,
    activation: ActivationSpec = "relu",
    dropout: float = 0.0,
    layer_norm: bool = False,
    output_activation: ActivationSpec | None = None,
    bias: bool = True,
) -> nn.Sequential:
    """Build a multi-layer perceptron.

    Each hidden layer is ``Linear -> [LayerNorm] -> activation -> [Dropout]``.
    When ``output_dim`` is given a final ``Linear`` layer is appended, optionally
    followed by ``output_activation``.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : sequence of int
        Widths of the hidden layers (may be empty).
    output_dim : int, optional
        Width of the output layer. ``None`` returns only the hidden stack, whose
        output width is ``hidden_dims[-1]`` (or ``input_dim`` if empty).
    activation : str or type, default ``'relu'``
        Hidden-layer activation, see :func:`get_activation`.
    dropout : float, default 0.0
        Dropout probability after each hidden activation (0 disables dropout).
    layer_norm : bool, default False
        Insert ``LayerNorm`` after each hidden ``Linear``.
    output_activation : str or type, optional
        Activation applied after the output layer; ``None`` keeps it linear.
    bias : bool, default True
        Whether the ``Linear`` layers have a bias.

    Returns
    -------
    torch.nn.Sequential

    Examples
    --------
    >>> build_mlp(3, [8], output_dim=1)
    Sequential(
      (0): Linear(in_features=3, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=1, bias=True)
    )
    """
    prev = _check_positive_int("input_dim", input_dim)
    hidden = _check_dims("hidden_dims", hidden_dims)
    dropout = _check_dropout(dropout)
    get_activation(activation)  # validate before building anything

    layers: list[nn.Module] = []
    for width in hidden:
        layers.append(nn.Linear(prev, width, bias=bias))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = width
    if output_dim is not None:
        layers.append(nn.Linear(prev, _check_positive_int("output_dim", output_dim), bias=bias))
        if output_activation is not None:
            layers.append(get_activation(output_activation))
    return nn.Sequential(*layers)


def _mlp_width(input_dim: int, hidden_dims: Sequence[int]) -> int:
    """Output width of a hidden stack built by :func:`build_mlp` without ``output_dim``."""
    return int(hidden_dims[-1]) if len(hidden_dims) else int(input_dim)


# ---------------------------------------------------------------------------
# Convolutional stacks
# ---------------------------------------------------------------------------


def build_conv_stack(
    in_channels: int,
    conv_dims: Sequence[int],
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    activation: ActivationSpec = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> nn.Sequential:
    """Build ``Conv2d -> [BatchNorm2d] -> activation -> [Dropout2d]`` blocks.

    Each convolution uses ``padding = kernel_size // 2`` ("same" padding for odd
    kernels at stride 1).

    Parameters
    ----------
    in_channels : int
        Channels of the input image.
    conv_dims, kernel_sizes, strides : sequence of int
        Output channels, kernel size and stride of each layer (equal lengths).
    activation : str or type, default ``'relu'``
        Activation after each convolution.
    dropout : float, default 0.0
        Channel dropout probability (``Dropout2d``).
    batch_norm : bool, default False
        Insert ``BatchNorm2d`` after each convolution.

    Returns
    -------
    torch.nn.Sequential
    """
    if not (len(conv_dims) == len(kernel_sizes) == len(strides)):
        raise ValueError(
            "conv_dims, kernel_sizes and strides must have the same length, got "
            f"{len(conv_dims)}, {len(kernel_sizes)} and {len(strides)}"
        )
    dropout = _check_dropout(dropout)
    layers: list[nn.Module] = []
    prev = _check_positive_int("in_channels", in_channels)
    for out_channels, kernel, stride in zip(conv_dims, kernel_sizes, strides):
        layers.append(nn.Conv2d(prev, out_channels, kernel, stride, padding=kernel // 2))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        prev = out_channels
    return nn.Sequential(*layers)


def _transposed_padding(kernel_size: int, stride: int) -> tuple[int, int]:
    """Padding and output padding so that a ``ConvTranspose2d`` scales H and W by ``stride``."""
    padding = max(0, math.ceil((kernel_size - stride) / 2))
    output_padding = 2 * padding + stride - kernel_size
    if not 0 <= output_padding < stride:
        raise ValueError(
            f"kernel_size={kernel_size} with stride={stride} cannot upsample by exactly "
            "the stride; use an odd kernel size or a stride >= 2"
        )
    return padding, output_padding


# ---------------------------------------------------------------------------
# Recurrent helpers
# ---------------------------------------------------------------------------


def build_rnn(
    rnn_type: str,
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    bidirectional: bool = False,
    dropout: float = 0.0,
) -> nn.RNNBase:
    """Build a batch-first ``nn.LSTM`` or ``nn.GRU``.

    Parameters
    ----------
    rnn_type : {'lstm', 'gru'}
        Recurrent cell type (case-insensitive).
    input_size, hidden_size, num_layers : int
        Standard PyTorch RNN sizes.
    bidirectional : bool, default False
        Use a bidirectional RNN.
    dropout : float, default 0.0
        Dropout between stacked layers (ignored when ``num_layers == 1``).

    Returns
    -------
    torch.nn.LSTM or torch.nn.GRU

    Raises
    ------
    ValueError
        If ``rnn_type`` is not one of :data:`RNN_TYPES`.
    """
    key = str(rnn_type).lower()
    if key not in RNN_TYPES:
        raise ValueError(
            f"Unsupported rnn_type {rnn_type!r}; valid types are: {', '.join(RNN_TYPES)}"
        )
    cls = nn.LSTM if key == "lstm" else nn.GRU
    return cls(
        _check_positive_int("input_size", input_size),
        _check_positive_int("hidden_size", hidden_size),
        _check_positive_int("num_layers", num_layers),
        bidirectional=bool(bidirectional),
        dropout=_check_dropout(dropout) if num_layers > 1 else 0.0,
        batch_first=True,
    )


def final_hidden_state(hidden: RNNHidden, bidirectional: bool) -> torch.Tensor:
    """Final hidden state of the last RNN layer, shape ``(batch, hidden * num_directions)``.

    For a bidirectional RNN this concatenates the forward direction's state after
    the last time step with the backward direction's state after the first time
    step, i.e. both directions have seen the whole sequence. (Taking
    ``output[:, -1]`` instead would give a backward state that saw one step only.)

    Parameters
    ----------
    hidden : Tensor or tuple of Tensor
        ``h_n`` (GRU) or ``(h_n, c_n)`` (LSTM) as returned by the RNN.
    bidirectional : bool
        Whether the RNN is bidirectional.
    """
    h_n = hidden[0] if isinstance(hidden, tuple) else hidden
    if bidirectional:
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)
    return h_n[-1]


# ---------------------------------------------------------------------------
# Transformer helpers
# ---------------------------------------------------------------------------


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """Sinusoidal position table of shape ``(max_len, d_model)`` (Vaswani et al., 2017).

    ``table[pos, 2i] = sin(pos / 10000^(2i / d_model))`` and
    ``table[pos, 2i + 1] = cos(pos / 10000^(2i / d_model))``. Odd ``d_model`` is supported.
    """
    max_len = _check_positive_int("max_len", max_len)
    d_model = _check_positive_int("d_model", d_model)
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    table = torch.zeros(max_len, d_model)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
    return table


class PositionalEncoding(nn.Module):
    """Additive sinusoidal positional encoding for batch-first sequences.

    Parameters
    ----------
    d_model : int
        Feature dimension of the sequence elements.
    dropout : float, default 0.1
        Dropout applied after adding the encoding.
    max_len : int, default 5000
        Longest supported sequence. The table is precomputed once and stored as
        a non-persistent buffer (it is not written to ``state_dict``).

    Notes
    -----
    Inputs have shape ``(batch, seq_len, d_model)``; position ``t`` receives
    ``pe[t]`` for every batch element.

    Raises
    ------
    ValueError
        In ``forward`` when ``seq_len > max_len`` or the feature size is not ``d_model``.

    Examples
    --------
    >>> pe = PositionalEncoding(16, dropout=0.0)
    >>> pe(torch.zeros(2, 5, 16)).shape
    torch.Size([2, 5, 16])
    """

    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = _check_positive_int("d_model", d_model)
        self.max_len = _check_positive_int("max_len", max_len)
        self.dropout = nn.Dropout(p=_check_dropout(dropout))
        self.register_buffer(
            "pe", sinusoidal_table(self.max_len, self.d_model).unsqueeze(0), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(
                f"PositionalEncoding expects input of shape (batch, seq_len, {self.d_model}), "
                f"got {tuple(x.shape)}"
            )
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds the positional encoding max_len={self.max_len}; "
                "construct the network with a larger max_len"
            )
        return self.dropout(x + self.pe[:, :seq_len].to(dtype=x.dtype))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_len={self.max_len}"


def build_transformer_encoder(
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float = 0.1,
) -> nn.TransformerEncoder:
    """Build a batch-first ``nn.TransformerEncoder`` (nested-tensor fast path disabled).

    Raises
    ------
    ValueError
        If ``d_model`` is not divisible by ``nhead``.
    """
    d_model = _check_positive_int("d_model", d_model)
    nhead = _check_positive_int("nhead", nhead)
    if d_model % nhead:
        raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=_check_positive_int("dim_feedforward", dim_feedforward),
        dropout=_check_dropout(dropout),
        activation="relu",
        batch_first=True,
    )
    return nn.TransformerEncoder(
        layer, _check_positive_int("num_layers", num_layers), enable_nested_tensor=False
    )


def masked_mean(x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over the sequence dimension of ``x`` (``(batch, seq, d)``), ignoring padding.

    Parameters
    ----------
    x : Tensor
        Sequence features, shape ``(batch, seq_len, d)``.
    padding_mask : Tensor, optional
        Boolean ``(batch, seq_len)`` mask where ``True`` marks padding (the PyTorch
        ``src_key_padding_mask`` convention). Fully padded rows yield zeros.

    Returns
    -------
    Tensor
        Shape ``(batch, d)``.
    """
    if padding_mask is None:
        return x.mean(dim=1)
    keep = (~padding_mask).unsqueeze(-1)
    total = x.masked_fill(~keep, 0.0).sum(dim=1)
    count = keep.sum(dim=1).clamp_min(1).to(x.dtype)
    return total / count


def _as_padding_mask(mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
    """Validate a ``(batch, seq_len)`` padding mask and convert it to bool."""
    if mask is None:
        return None
    if mask.shape != x.shape[:2]:
        raise ValueError(
            f"mask must have shape (batch, seq_len) = {tuple(x.shape[:2])}, got {tuple(mask.shape)}"
        )
    return mask.to(device=x.device, dtype=torch.bool)
