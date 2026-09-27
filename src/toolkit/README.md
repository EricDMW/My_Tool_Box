# Toolkit: A Comprehensive Research Toolkit

A comprehensive research toolkit for reinforcement learning and research applications, providing both neural network architectures and plotting utilities.

## Components

### Neural Toolkit (`neural_toolkit`)

A flexible neural network toolkit designed for reinforcement learning applications, providing:

- **Policy Networks**: MLP, CNN, RNN, and Transformer-based policy networks
- **Value Networks**: Various value function approximators
- **Q-Networks**: Q-function networks including Dueling architectures
- **Encoders/Decoders**: Flexible encoding and decoding architectures
- **Discrete Tools**: Tools for discrete action spaces and tabular methods
- **Utilities**: Network utilities and helper functions

### Plotkit (`plotkit`)

A research-quality plotting toolkit providing:

- **Research-Style Plots**: Publication-ready plotting functions
- **Specialized Visualizations**: Shadow curves, heatmaps, gray scales
- **Standard Plots**: Line, bar, and scatter plots with research styling
- **Color Schemes**: Predefined research color palettes

## Installation

```bash
pip install toolkit
```

## Quick Start

```python
import toolkit

# Use neural toolkit
from toolkit.neural_toolkit import MLPPolicyNetwork, MLPValueNetwork
from toolkit.neural_toolkit import NetworkUtils

# Use plotkit
from toolkit.plotkit import plot_shadow_curve, plot_heatmap, set_research_style
from toolkit.plotkit import RESEARCH_COLORS

# Set research style for all plots
set_research_style()
```

## Dependencies

- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- numpy >= 1.20.0
- torch >= 1.9.0
- tensorflow >= 2.6.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0

## License

MIT License

## Author

Dongming Wang 