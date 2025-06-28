"""
Neural Toolkit for Reinforcement Learning

A comprehensive toolkit providing flexible neural network architectures and utilities
for reinforcement learning applications.
"""

# Import network architectures
from .networks.policy_networks import (
    BasePolicyNetwork,
    MLPPolicyNetwork,
    CNPolicyNetwork,
    RNPolicyNetwork,
    TransformerPolicyNetwork,
    PolicyFactory
)

from .networks.value_networks import (
    BaseValueNetwork,
    MLPValueNetwork,
    CNNValueNetwork,
    RNNValueNetwork,
    TransformerValueNetwork,
    ValueFactory
)

from .networks.q_networks import (
    BaseQNetwork,
    MLPQNetwork,
    DuelingQNetwork,
    CNQNetwork,
    RNQNetwork,
    TransformerQNetwork,
    QNetworkFactory
)

# Import encoders and decoders
from .encoders.encoders import (
    BaseEncoder,
    MLPEncoder,
    CNNEncoder,
    RNNEncoder,
    TransformerEncoder,
    VariationalEncoder,
    EncoderFactory
)

from .decoders.decoders import (
    BaseDecoder,
    MLPDecoder,
    CNNDecoder,
    RNNDecoder,
    TransformerDecoder,
    VariationalDecoder,
    DecoderFactory
)

# Import discrete tools
from .discrete_tools.discrete_tools import (
    BaseDiscreteTable,
    QTable,
    ValueTable,
    PolicyTable,
    DiscreteTools,
    DiscreteEnvironment
)

# Import utilities
from .utils.network_utils import NetworkUtils

# Version information
__version__ = "1.0.0"
__author__ = "Neural Toolkit Team"

# Main classes for easy access
__all__ = [
    # Policy Networks
    "BasePolicyNetwork",
    "MLPPolicyNetwork", 
    "CNPolicyNetwork",
    "RNPolicyNetwork",
    "TransformerPolicyNetwork",
    "PolicyFactory",
    
    # Value Networks
    "BaseValueNetwork",
    "MLPValueNetwork",
    "CNNValueNetwork", 
    "RNNValueNetwork",
    "TransformerValueNetwork",
    "ValueFactory",
    
    # Q Networks
    "BaseQNetwork",
    "MLPQNetwork",
    "DuelingQNetwork",
    "CNQNetwork",
    "RNQNetwork", 
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
    
    # Discrete Tools
    "BaseDiscreteTable",
    "QTable",
    "ValueTable",
    "PolicyTable",
    "DiscreteTools",
    "DiscreteEnvironment",
    
    # Utilities
    "NetworkUtils"
] 