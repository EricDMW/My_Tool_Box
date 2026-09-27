"""Action-value networks ``Q(s, a)``.

Discrete Q-networks map a state to one value per action, shape
``(batch, action_dim)``. Passing integer ``action`` indices to ``forward``
returns the selected values ``Q(s, a)`` with shape ``(batch,)`` instead, which is
what TD losses need.

:class:`MLPQNetwork` also supports continuous actions (``continuous_action=True``):
state and action are concatenated and the network returns ``Q(s, a)`` with shape
``(batch, 1)``, as used by DDPG/TD3/SAC critics.

=======================  ===============================  ===================
Class                    State shape                      Output shape
=======================  ===============================  ===================
MLPQNetwork              ``(batch, state_dim)``           ``(batch, action_dim)``
DuelingQNetwork          ``(batch, state_dim)``           ``(batch, action_dim)``
CNQNetwork               ``(batch, C, H, W)``             ``(batch, action_dim)``
RNQNetwork               ``(batch, seq_len, state_dim)``  ``((batch, action_dim), hidden)``
TransformerQNetwork      ``(batch, seq_len, state_dim)``  ``(batch, action_dim)``
=======================  ===============================  ===================

``CNNQNetwork`` and ``RNNQNetwork`` are aliases of the CNN and RNN classes.
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
    "BaseQNetwork",
    "CNNQNetwork",
    "CNQNetwork",
    "DuelingQNetwork",
    "MLPQNetwork",
    "PositionalEncoding",
    "QNetworkFactory",
    "RNNQNetwork",
    "RNQNetwork",
    "TransformerQNetwork",
]


def select_action_values(q_values: torch.Tensor, action: torch.Tensor | None) -> torch.Tensor:
    """Return ``q_values`` or, when ``action`` is given, ``Q(s, a)`` of shape ``(batch,)``.

    Parameters
    ----------
    q_values : Tensor
        Shape ``(batch, action_dim)``.
    action : Tensor, optional
        Integer action indices of shape ``(batch,)`` or ``(batch, 1)``.
    """
    if action is None:
        return q_values
    index = action.to(device=q_values.device, dtype=torch.long)
    if index.dim() == q_values.dim():
        index = index.squeeze(-1)
    if index.shape != q_values.shape[:-1]:
        raise ValueError(
            f"action must have shape {tuple(q_values.shape[:-1])} (one index per state), "
            f"got {tuple(action.shape)}"
        )
    return q_values.gather(-1, index.unsqueeze(-1)).squeeze(-1)


class BaseQNetwork(nn.Module, ABC):
    """Base class for Q-networks.

    Parameters
    ----------
    state_dim : int
        State feature dimension.
    action_dim : int
        Number of discrete actions (or the action dimension for continuous critics).
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
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = _check_positive_int("action_dim", action_dim)
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
    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None):
        """Forward pass; see the concrete subclasses for shapes."""


class MLPQNetwork(BaseQNetwork):
    """Multi-layer perceptron Q-network.

    Parameters
    ----------
    state_dim, action_dim, hidden_dims, activation, dropout, layer_norm, device
        See :class:`BaseQNetwork`.
    continuous_action : bool, default False
        If True the network is a continuous-action critic: ``forward(state, action)``
        concatenates ``state`` and ``action`` (shape ``(batch, action_dim)``) and
        returns ``Q(s, a)`` of shape ``(batch, 1)``.

    Examples
    --------
    >>> q = MLPQNetwork(state_dim=6, action_dim=3)
    >>> q(torch.randn(5, 6)).shape
    torch.Size([5, 3])
    >>> q(torch.randn(5, 6), torch.tensor([0, 1, 2, 1, 0])).shape
    torch.Size([5])
    >>> critic = MLPQNetwork(state_dim=6, action_dim=2, continuous_action=True)
    >>> critic(torch.randn(5, 6), torch.randn(5, 2)).shape
    torch.Size([5, 1])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: ActivationSpec = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        device: Device = "cpu",
        continuous_action: bool = False,
    ) -> None:
        self.continuous_action = bool(continuous_action)
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_network(self) -> None:
        self.state_dim = _check_positive_int("state_dim", self.state_dim)
        in_dim = self.state_dim + (self.action_dim if self.continuous_action else 0)
        self.state_layers = build_mlp(
            in_dim,
            self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )
        out_dim = 1 if self.continuous_action else self.action_dim
        self.q_layer = nn.Linear(_mlp_width(in_dim, self.hidden_dims), out_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Return Q-values; see the class docstring for the output shapes."""
        if self.continuous_action:
            if action is None:
                raise ValueError("MLPQNetwork(continuous_action=True) requires an action tensor")
            if action.shape[:-1] != state.shape[:-1] or action.size(-1) != self.action_dim:
                raise ValueError(
                    f"action must have shape {(*state.shape[:-1], self.action_dim)}, "
                    f"got {tuple(action.shape)}"
                )
            x = torch.cat([state, action.to(state.dtype)], dim=-1)
            return self.q_layer(self.state_layers(x))
        return select_action_values(self.q_layer(self.state_layers(state)), action)


class DuelingQNetwork(BaseQNetwork):
    """Dueling Q-network (Wang et al., 2016).

    The shared trunk uses ``hidden_dims[:-1]``; the value and advantage streams
    each have one hidden layer of width ``hidden_dims[-1]``. The streams are
    combined as ``Q = V + A - mean_a(A)``.

    Raises
    ------
    ValueError
        If ``hidden_dims`` is empty.
    """

    def _build_network(self) -> None:
        self.state_dim = _check_positive_int("state_dim", self.state_dim)
        if not self.hidden_dims:
            raise ValueError("DuelingQNetwork needs at least one entry in hidden_dims")
        shared_dims, stream_dim = self.hidden_dims[:-1], self.hidden_dims[-1]
        mlp_kwargs = dict(
            activation=self.activation, dropout=self.dropout, layer_norm=self.layer_norm
        )
        self.shared_layers = build_mlp(self.state_dim, shared_dims, **mlp_kwargs)
        shared_out = _mlp_width(self.state_dim, shared_dims)
        self.value_layers = build_mlp(shared_out, [stream_dim], output_dim=1, **mlp_kwargs)
        self.advantage_layers = build_mlp(
            shared_out, [stream_dim], output_dim=self.action_dim, **mlp_kwargs
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Return Q-values ``(batch, action_dim)``, or ``Q(s, a)`` ``(batch,)`` if ``action`` is given."""
        shared = self.shared_layers(state)
        value = self.value_layers(shared)
        advantage = self.advantage_layers(shared)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return select_action_values(q_values, action)


class CNQNetwork(_ConvTrunk, BaseQNetwork):
    """Convolutional Q-network for image states.

    Parameters
    ----------
    input_channels : int
        Image channels of states shaped ``(batch, C, H, W)``.
    action_dim : int
        Number of discrete actions.
    conv_dims, fc_dims, kernel_sizes, strides, activation, dropout, layer_norm, device
        See :class:`~toolkit.neural_toolkit.CNPolicyNetwork`.
    """

    def __init__(
        self,
        input_channels: int,
        action_dim: int,
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
            state_dim=0,
            action_dim=action_dim,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_network(self) -> None:
        self.q_layer = nn.Linear(self._build_conv_trunk(), self.action_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Return Q-values ``(batch, action_dim)``, or ``Q(s, a)`` ``(batch,)`` if ``action`` is given."""
        return select_action_values(self.q_layer(self._conv_features(state)), action)


class RNQNetwork(_RecurrentTrunk, BaseQNetwork):
    """Recurrent (LSTM/GRU) Q-network, e.g. for DRQN.

    Parameters are those of :class:`~toolkit.neural_toolkit.RNPolicyNetwork` with
    ``state_dim``/``action_dim`` in place of ``input_dim``/``output_dim``.
    ``forward(state, action=None, hidden=None)`` returns ``(q_values, hidden)``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
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
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_network(self) -> None:
        self.state_dim = _check_positive_int("state_dim", self.state_dim)
        self.q_layer = nn.Linear(self._build_recurrent_trunk(self.state_dim), self.action_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        hidden: RNNHidden | None = None,
    ) -> tuple[torch.Tensor, RNNHidden]:
        """Return ``(q_values, hidden)``; ``q_values`` as in :class:`MLPQNetwork`."""
        features, hidden = self._recurrent_features(state, hidden)
        return select_action_values(self.q_layer(features), action), hidden


class TransformerQNetwork(_TransformerTrunk, BaseQNetwork):
    """Transformer-encoder Q-network.

    Parameters are those of :class:`~toolkit.neural_toolkit.TransformerPolicyNetwork`
    with ``state_dim``/``action_dim`` in place of ``input_dim``/``output_dim``.
    ``forward(state, action=None, mask=None)`` accepts a boolean padding mask
    ``(batch, seq_len)`` (``True`` = padding).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
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
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=fc_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            device=device,
        )

    def _build_network(self) -> None:
        self.state_dim = _check_positive_int("state_dim", self.state_dim)
        self.q_layer = nn.Linear(self._build_transformer_trunk(self.state_dim), self.action_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return Q-values ``(batch, action_dim)``, or ``Q(s, a)`` ``(batch,)`` if ``action`` is given."""
        return select_action_values(self.q_layer(self._transformer_features(state, mask)), action)


CNNQNetwork = CNQNetwork
RNNQNetwork = RNQNetwork


class QNetworkFactory:
    """Create Q-networks by name.

    Valid types: ``'mlp'``, ``'dueling'``, ``'cnn'``, ``'rnn'``, ``'transformer'``.
    """

    NETWORK_TYPES: dict[str, type[BaseQNetwork]] = {
        "mlp": MLPQNetwork,
        "dueling": DuelingQNetwork,
        "cnn": CNQNetwork,
        "rnn": RNQNetwork,
        "transformer": TransformerQNetwork,
    }

    @staticmethod
    def create_q_network(network_type: str, **kwargs) -> BaseQNetwork:
        """Instantiate the Q-network registered under ``network_type``.

        Raises
        ------
        ValueError
            If ``network_type`` is unknown.
        """
        key = str(network_type).lower()
        if key not in QNetworkFactory.NETWORK_TYPES:
            valid = ", ".join(QNetworkFactory.NETWORK_TYPES)
            raise ValueError(f"Unsupported network type {network_type!r}; valid types are: {valid}")
        return QNetworkFactory.NETWORK_TYPES[key](**kwargs)
