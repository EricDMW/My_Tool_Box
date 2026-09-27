"""Neural toolkit: configurable PyTorch building blocks for reinforcement learning.

Contents
--------
Networks
    Policy, value and Q networks with MLP, CNN, RNN (LSTM/GRU) and transformer
    trunks, plus factories that build them by name.
Encoders and decoders
    MLP, CNN, RNN, transformer and variational (VAE) encoders/decoders.
Tabular tools
    Q-tables, value tables, policy tables and classic tabular updates.
Layers
    Shared primitives: :func:`get_activation`, :func:`build_mlp` and
    :class:`PositionalEncoding`.
Utilities
    :class:`NetworkUtils` (initialisation, optimisers, schedulers, checkpoints).

Every MLP in the toolkit is ``Linear -> [LayerNorm] -> activation -> [Dropout]``
per hidden layer with a linear output layer. Constructors accept a ``device``
argument and move the module there at the end of construction.

Requires PyTorch (``pip install "my-tool-box[torch]"``).
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "toolkit.neural_toolkit requires PyTorch. Install it with "
        "'pip install \"my-tool-box[torch]\"' or follow https://pytorch.org/get-started/."
    ) from exc

from toolkit._version import __version__

from .decoders.decoders import (
    BaseDecoder,
    CNNDecoder,
    DecoderFactory,
    MLPDecoder,
    RNNDecoder,
    TransformerDecoder,
    VariationalDecoder,
)
from .discrete_tools.discrete_tools import (
    BaseDiscreteTable,
    DiscreteEnvironment,
    DiscreteTools,
    PolicyTable,
    QTable,
    ValueTable,
)
from .encoders.encoders import (
    BaseEncoder,
    CNNEncoder,
    EncoderFactory,
    MLPEncoder,
    RNNEncoder,
    TransformerEncoder,
    VariationalEncoder,
)
from .layers import ACTIVATION_NAMES, PositionalEncoding, build_mlp, get_activation
from .networks.policy_networks import (
    BasePolicyNetwork,
    CNNPolicyNetwork,
    CNPolicyNetwork,
    MLPPolicyNetwork,
    PolicyFactory,
    RNNPolicyNetwork,
    RNPolicyNetwork,
    TransformerPolicyNetwork,
)
from .networks.q_networks import (
    BaseQNetwork,
    CNNQNetwork,
    CNQNetwork,
    DuelingQNetwork,
    MLPQNetwork,
    QNetworkFactory,
    RNNQNetwork,
    RNQNetwork,
    TransformerQNetwork,
)
from .networks.value_networks import (
    BaseValueNetwork,
    CNNValueNetwork,
    MLPValueNetwork,
    RNNValueNetwork,
    TransformerValueNetwork,
    ValueFactory,
)
from .utils.network_utils import NetworkUtils

__author__ = "Dongming Wang"

__all__ = [
    "__version__",
    # Policy networks
    "BasePolicyNetwork",
    "MLPPolicyNetwork",
    "CNPolicyNetwork",
    "CNNPolicyNetwork",
    "RNPolicyNetwork",
    "RNNPolicyNetwork",
    "TransformerPolicyNetwork",
    "PolicyFactory",
    # Value networks
    "BaseValueNetwork",
    "MLPValueNetwork",
    "CNNValueNetwork",
    "RNNValueNetwork",
    "TransformerValueNetwork",
    "ValueFactory",
    # Q networks
    "BaseQNetwork",
    "MLPQNetwork",
    "DuelingQNetwork",
    "CNQNetwork",
    "CNNQNetwork",
    "RNQNetwork",
    "RNNQNetwork",
    "TransformerQNetwork",
    "QNetworkFactory",
    # Encoders
    "BaseEncoder",
    "MLPEncoder",
    "CNNEncoder",
    "RNNEncoder",
    "TransformerEncoder",
    "VariationalEncoder",
    "EncoderFactory",
    # Decoders
    "BaseDecoder",
    "MLPDecoder",
    "CNNDecoder",
    "RNNDecoder",
    "TransformerDecoder",
    "VariationalDecoder",
    "DecoderFactory",
    # Tabular tools
    "BaseDiscreteTable",
    "QTable",
    "ValueTable",
    "PolicyTable",
    "DiscreteTools",
    "DiscreteEnvironment",
    # Layers
    "ACTIVATION_NAMES",
    "PositionalEncoding",
    "build_mlp",
    "get_activation",
    # Utilities
    "NetworkUtils",
]
