"""Utility functions for building, initialising, optimising and checkpointing networks.

All helpers are static methods of :class:`NetworkUtils`.

Examples
--------
>>> import torch.nn as nn
>>> from toolkit.neural_toolkit import NetworkUtils
>>> model = NetworkUtils.create_mlp_layers(4, [32, 32], 2)
>>> NetworkUtils.initialize_weights(model, method="orthogonal")
>>> opt = NetworkUtils.create_optimizer(model, "adamw", lr=1e-3, weight_decay=1e-2)
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn

from ..layers import (
    ActivationSpec,
    _check_positive_int,
    build_conv_stack,
    build_mlp,
    get_activation,
    sinusoidal_table,
)

__all__ = ["NetworkUtils"]

PathLike = Union[str, os.PathLike]

_INIT_LAYERS = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)
_INIT_METHODS = (
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
    "uniform",
    "normal",
)
_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
    "adamax": torch.optim.Adamax,
}
_SCHEDULERS = {
    "step": torch.optim.lr_scheduler.StepLR,
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cosine_warm_restart": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "linear": torch.optim.lr_scheduler.LinearLR,
    "polynomial": torch.optim.lr_scheduler.PolynomialLR,
}


def _lookup(kind: str, table: dict[str, Any], name: str) -> Any:
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unsupported {kind} {name!r}; valid options are: {', '.join(table)}")
    return table[key]


def _per_layer_list(
    name: str, value: Sequence[int] | None, n: int, default: list[int]
) -> list[int]:
    values = list(default) if value is None else list(value)
    if len(values) != n:
        raise ValueError(f"{name} must have {n} entries (one per layer), got {len(values)}")
    return values


def _create_temp_file(target: Path) -> str:
    """Create an empty, uniquely named temporary file next to ``target``.

    Unlike ``tempfile.mkstemp`` (mode 0600), the file is created with mode 0666
    filtered by the process umask, so the renamed checkpoint has the permissions of
    a file written with :func:`open`. The umask itself is not read or changed.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        name = target.parent / f".{target.name}.{secrets.token_hex(6)}.tmp"
        try:
            os.close(os.open(name, flags, 0o666))
        except FileExistsError:
            continue
        return str(name)
    raise FileExistsError(f"could not create a temporary file next to {target}")


class NetworkUtils:
    """Static helpers for common neural-network chores."""

    # ------------------------------------------------------------------ init

    @staticmethod
    def initialize_weights(
        module: nn.Module, method: str = "xavier_uniform", gain: float = 1.0
    ) -> None:
        """Initialise the weights of every linear/convolutional layer in ``module``.

        The module is traversed recursively (``module.modules()``), so passing a
        whole network works. Biases are set to zero; other layer types are left
        untouched.

        Parameters
        ----------
        module : torch.nn.Module
            Network or single layer.
        method : str, default ``'xavier_uniform'``
            One of ``'xavier_uniform'``, ``'xavier_normal'``, ``'kaiming_uniform'``,
            ``'kaiming_normal'``, ``'orthogonal'``, ``'uniform'`` (U(-0.1, 0.1)) or
            ``'normal'`` (N(0, 0.1^2)).
        gain : float, default 1.0
            Gain for the Xavier and orthogonal initialisers.

        Raises
        ------
        ValueError
            If ``method`` is unknown.
        """
        if method not in _INIT_METHODS:
            raise ValueError(
                f"Unsupported initialization method {method!r}; valid methods are: "
                f"{', '.join(_INIT_METHODS)}"
            )
        for layer in module.modules():
            if not isinstance(layer, _INIT_LAYERS):
                continue
            weight = layer.weight
            if method == "xavier_uniform":
                nn.init.xavier_uniform_(weight, gain=gain)
            elif method == "xavier_normal":
                nn.init.xavier_normal_(weight, gain=gain)
            elif method == "kaiming_uniform":
                nn.init.kaiming_uniform_(weight, mode="fan_in", nonlinearity="relu")
            elif method == "kaiming_normal":
                nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="relu")
            elif method == "orthogonal":
                nn.init.orthogonal_(weight, gain=gain)
            elif method == "uniform":
                nn.init.uniform_(weight, -0.1, 0.1)
            else:
                nn.init.normal_(weight, 0.0, 0.1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------ parameters

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def count_parameters_by_layer(model: nn.Module) -> dict[str, int]:
        """Trainable parameters owned directly by each sub-module (the root is ``''``).

        Only modules that own at least one trainable parameter are listed, so the
        values sum to :meth:`count_parameters`.
        """
        counts: dict[str, int] = {}
        for name, module in model.named_modules():
            n = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            if n:
                counts[name] = n
        return counts

    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: Iterable[str]) -> None:
        """Disable gradients of parameters whose name contains any of ``layer_names``."""
        names = list(layer_names)
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in names):
                param.requires_grad = False

    @staticmethod
    def unfreeze_layers(model: nn.Module, layer_names: Iterable[str]) -> None:
        """Enable gradients of parameters whose name contains any of ``layer_names``."""
        names = list(layer_names)
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in names):
                param.requires_grad = True

    @staticmethod
    def get_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
        """Total gradient norm over all parameters (``inf`` gives the max-abs norm).

        Returns 0.0 when no parameter has a gradient.
        """
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        norms = torch.stack([torch.linalg.vector_norm(g, norm_type) for g in grads])
        return float(torch.linalg.vector_norm(norms, norm_type))

    @staticmethod
    def clip_grad_norm(model: nn.Module, max_norm: float, norm_type: float = 2.0) -> float:
        """Clip gradients in place and return the total norm before clipping."""
        return float(nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type))

    # --------------------------------------------------------------- builders

    @staticmethod
    def get_activation(activation: ActivationSpec) -> nn.Module:
        """Activation module by name; see :func:`toolkit.neural_toolkit.layers.get_activation`.

        Raises
        ------
        ValueError
            For unknown names (the lookup no longer falls back to ReLU).
        """
        return get_activation(activation)

    @staticmethod
    def create_mlp_layers(
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        bias: bool = True,
    ) -> nn.Sequential:
        """MLP ``Linear -> [LayerNorm] -> activation -> [Dropout]`` per hidden layer, linear output.

        See :func:`toolkit.neural_toolkit.layers.build_mlp`.
        """
        return build_mlp(
            input_dim,
            hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            bias=bias,
        )

    @staticmethod
    def create_conv_layers(
        input_channels: int,
        conv_dims: Sequence[int],
        kernel_sizes: Sequence[int] | None = None,
        strides: Sequence[int] | None = None,
        padding: Sequence[int] | None = None,
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> nn.Sequential:
        """``Conv2d -> [BatchNorm2d] -> activation -> [Dropout2d]`` blocks.

        Defaults: kernel size 3, stride 1, padding ``kernel_size // 2``. Every
        per-layer list must have one entry per element of ``conv_dims``.
        """
        n = len(conv_dims)
        kernel_sizes = _per_layer_list("kernel_sizes", kernel_sizes, n, [3] * n)
        strides = _per_layer_list("strides", strides, n, [1] * n)
        if padding is None:
            return build_conv_stack(
                input_channels, conv_dims, kernel_sizes, strides, activation, dropout, layer_norm
            )
        padding = _per_layer_list("padding", padding, n, [])
        layers: list[nn.Module] = []
        prev = _check_positive_int("input_channels", input_channels)
        for out_channels, kernel, stride, pad in zip(conv_dims, kernel_sizes, strides, padding):
            layers.append(nn.Conv2d(prev, out_channels, kernel, stride, pad))
            if layer_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            prev = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def create_transposed_conv_layers(
        input_channels: int,
        conv_dims: Sequence[int],
        kernel_sizes: Sequence[int] | None = None,
        strides: Sequence[int] | None = None,
        padding: Sequence[int] | None = None,
        output_padding: Sequence[int] | None = None,
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> nn.Sequential:
        """``ConvTranspose2d -> [BatchNorm2d] -> activation -> [Dropout2d]`` blocks.

        Defaults: kernel size 3, stride 2, padding ``kernel_size // 2`` and
        output padding ``stride - 1`` (doubling H and W for odd kernels).
        """
        n = len(conv_dims)
        kernel_sizes = _per_layer_list("kernel_sizes", kernel_sizes, n, [3] * n)
        strides = _per_layer_list("strides", strides, n, [2] * n)
        padding = _per_layer_list("padding", padding, n, [k // 2 for k in kernel_sizes])
        output_padding = _per_layer_list(
            "output_padding", output_padding, n, [s - 1 for s in strides]
        )
        layers: list[nn.Module] = []
        prev = _check_positive_int("input_channels", input_channels)
        for out_channels, kernel, stride, pad, out_pad in zip(
            conv_dims, kernel_sizes, strides, padding, output_padding
        ):
            layers.append(nn.ConvTranspose2d(prev, out_channels, kernel, stride, pad, out_pad))
            if layer_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            prev = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def create_attention_layer(
        input_dim: int, num_heads: int = 8, dropout: float = 0.1
    ) -> nn.Module:
        """Batch-first ``nn.MultiheadAttention``.

        Raises
        ------
        ValueError
            If ``input_dim`` is not divisible by ``num_heads``.
        """
        if input_dim % num_heads:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
            )
        return nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)

    @staticmethod
    def create_positional_encoding(d_model: int, max_len: int = 5000) -> nn.Parameter:
        """Frozen sinusoidal table of shape ``(1, max_len, d_model)`` for batch-first inputs.

        Add ``pe[:, :seq_len]`` to a ``(batch, seq_len, d_model)`` tensor. For a
        module that also validates the length, use
        :class:`toolkit.neural_toolkit.layers.PositionalEncoding`.
        """
        return nn.Parameter(sinusoidal_table(max_len, d_model).unsqueeze(0), requires_grad=False)

    # ------------------------------------------------------------- optimisers

    @staticmethod
    def apply_weight_decay(
        model: nn.Module, weight_decay: float, skip_list: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        """Optimizer parameter groups with weight decay on weights only.

        One-dimensional parameters (biases, normalisation scales) and parameters
        whose name contains an entry of ``skip_list`` get ``weight_decay=0``.
        Frozen parameters are excluded and empty groups are dropped.

        Returns
        -------
        list of dict
            Parameter groups for a ``torch.optim.Optimizer``.
        """
        skip = list(skip_list or [])
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or any(s in name for s in skip):
                no_decay.append(param)
            else:
                decay.append(param)
        groups = [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return [g for g in groups if g["params"]]

    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optimizer_type: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        skip_list: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        """Create an optimizer; with ``weight_decay > 0`` uses :meth:`apply_weight_decay` groups.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose trainable parameters are optimised.
        optimizer_type : str, default ``'adam'``
            One of ``'adam'``, ``'adamw'``, ``'sgd'``, ``'rmsprop'``, ``'adagrad'``, ``'adamax'``.
        lr : float, default 1e-3
            Learning rate.
        weight_decay : float, default 0.0
            Decay applied to weight matrices (not to biases or norm scales).
        skip_list : iterable of str, optional
            Name fragments of parameters excluded from weight decay.
        **kwargs
            Extra optimizer arguments. SGD defaults to ``momentum=0.9``.

        Raises
        ------
        ValueError
            If ``optimizer_type`` is unknown.
        """
        optimizer_class = _lookup("optimizer type", _OPTIMIZERS, optimizer_type)
        if weight_decay > 0:
            params: Any = NetworkUtils.apply_weight_decay(model, weight_decay, skip_list)
        else:
            params = [p for p in model.parameters() if p.requires_grad]
        if optimizer_class is torch.optim.SGD:
            kwargs.setdefault("momentum", 0.9)
        return optimizer_class(params, lr=lr, weight_decay=weight_decay, **kwargs)

    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer, scheduler_type: str = "step", **kwargs: Any
    ) -> Any:
        """Create a learning-rate scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to schedule.
        scheduler_type : str, default ``'step'``
            One of ``'step'``, ``'multistep'``, ``'exponential'``, ``'cosine'``,
            ``'cosine_warm_restart'``, ``'plateau'``, ``'linear'``, ``'polynomial'``.
        **kwargs
            Arguments of the scheduler class (for example ``step_size`` or ``T_max``).

        Raises
        ------
        ValueError
            If ``scheduler_type`` is unknown.
        """
        scheduler_class = _lookup("scheduler type", _SCHEDULERS, scheduler_type)
        return scheduler_class(optimizer, **kwargs)

    # ------------------------------------------------------------ checkpoints

    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        loss: float,
        filepath: PathLike,
        **extra: Any,
    ) -> Path:
        """Save model/optimizer state, epoch and loss to ``filepath``.

        Parent directories are created and the file is written atomically; it gets
        the default permissions of new files (``0o666`` minus the umask).

        Parameters
        ----------
        model : torch.nn.Module
            Model whose ``state_dict`` is saved.
        optimizer : torch.optim.Optimizer, optional
            Optimizer whose ``state_dict`` is saved (``None`` stores ``None``).
        epoch : int
            Epoch number stored as ``"epoch"``.
        loss : float
            Loss value stored as ``"loss"``.
        filepath : str or path-like
            Destination file.
        **extra
            Additional entries stored under their names. :meth:`load_checkpoint`
            uses ``weights_only=True`` by default, which only accepts tensors and
            plain Python values (numbers, strings, lists, tuples, dicts of them);
            extras holding other objects (e.g. NumPy arrays or scalars, paths,
            custom classes) require ``load_checkpoint(..., weights_only=False)``,
            which should only be used for trusted files.

        Returns
        -------
        pathlib.Path
            The written file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "loss": float(loss),
            **extra,
        }
        tmp_name = _create_temp_file(path)
        try:
            torch.save(payload, tmp_name)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        return path

    @staticmethod
    def load_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        filepath: PathLike,
        map_location: str | torch.device | None = None,
        weights_only: bool = True,
        strict: bool = True,
    ) -> tuple[int, float]:
        """Restore a checkpoint written by :meth:`save_checkpoint`.

        Parameters
        ----------
        model : torch.nn.Module
            Model receiving the saved state.
        optimizer : torch.optim.Optimizer, optional
            Optimizer receiving the saved state (skipped if ``None`` or not saved).
        filepath : str or path-like
            Checkpoint file.
        map_location : str or torch.device, optional
            Where to load tensors. Defaults to the device of ``model``'s first
            parameter (CPU for parameterless models), so GPU checkpoints load on
            CPU-only machines.
        weights_only : bool, default True
            Passed to :func:`torch.load`; set False only for trusted files that
            contain arbitrary Python objects.
        strict : bool, default True
            Passed to ``model.load_state_dict``.

        Returns
        -------
        tuple of (int, float)
            ``(epoch, loss)``.
        """
        if map_location is None:
            first = next(model.parameters(), None)
            map_location = first.device if first is not None else torch.device("cpu")
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=weights_only)
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint["epoch"]), float(checkpoint["loss"])

    # ------------------------------------------------------------- shape math

    @staticmethod
    def compute_output_size(
        input_size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
    ) -> int:
        """Output length of a convolution: ``floor((n + 2p - d(k - 1) - 1) / s) + 1``.

        Raises
        ------
        ValueError
            If the result is not positive (kernel larger than the padded input).
        """
        out = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        if out <= 0:
            raise ValueError(
                f"Convolution with kernel_size={kernel_size}, stride={stride}, padding={padding}, "
                f"dilation={dilation} produces an empty output for input size {input_size}"
            )
        return out

    @staticmethod
    def compute_transposed_output_size(
        input_size: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
    ) -> int:
        """Output length of a transposed convolution: ``(n - 1)s - 2p + d(k - 1) + op + 1``."""
        return (
            (input_size - 1) * stride
            - 2 * padding
            + dilation * (kernel_size - 1)
            + output_padding
            + 1
        )

    @staticmethod
    def compute_conv_output_size(
        input_size: tuple[int, int], conv_layers: Sequence[dict[str, int]]
    ) -> tuple[int, int]:
        """Spatial size ``(H, W)`` after a stack of convolutions.

        Each entry of ``conv_layers`` may define ``kernel_size`` (default 3),
        ``stride`` (1), ``padding`` (``kernel_size // 2``) and ``dilation`` (1).
        """
        h, w = input_size
        for cfg in conv_layers:
            kernel = cfg.get("kernel_size", 3)
            stride = cfg.get("stride", 1)
            padding = cfg.get("padding", kernel // 2)
            dilation = cfg.get("dilation", 1)
            h = NetworkUtils.compute_output_size(h, kernel, stride, padding, dilation)
            w = NetworkUtils.compute_output_size(w, kernel, stride, padding, dilation)
        return h, w
