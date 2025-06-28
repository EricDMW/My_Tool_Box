"""
Toolkit: A comprehensive research toolkit for reinforcement learning and research applications.

This package contains two main components:
- neural_toolkit: Neural network architectures and utilities for RL
- plotkit: Research-quality plotting and visualization tools
"""

# Import neural toolkit components
from . import neural_toolkit
from . import plotkit

# Version information
__version__ = "0.2.0"
__author__ = "Dongming Wang"

# Make subpackages available at the top level
__all__ = [
    "neural_toolkit",
    "plotkit"
] 