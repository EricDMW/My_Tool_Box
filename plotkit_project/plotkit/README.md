# plotkit

A simple, extensible plotting toolkit for reinforcement learning and research.

## Features
- Shadow curves (mean ± std or confidence interval)
- Heatmaps
- Line plots
- Bar plots
- Scatter plots
- Accepts numpy arrays, PyTorch tensors, or TensorFlow tensors as input
- Easy-to-use API, built on matplotlib and seaborn

## Installation

```bash
pip install -e .
```
(from the plotkit directory)

## Usage

```python
import numpy as np
import plotkit
import matplotlib.pyplot as plt

# Shadow curve (1D input)
x = np.arange(100)
y = np.sin(x / 10)
y_std = 0.1 + 0.1 * np.abs(np.cos(x / 10))
plotkit.plot_shadow_curve(x, y, y_std, label='sin(x)', color='blue')
plt.legend()
plt.show()

# Shadow curve (2D input, axis selection, e.g. for RL runs)
import torch
runs = 5
steps = 100
y_tensor = torch.sin(torch.arange(steps).float() / 10) + 0.1 * torch.randn(runs, steps)
# Average over runs (axis=0)
plotkit.plot_shadow_curve(y_tensor, axis=0, label='mean over runs')
plt.legend()
plt.show()

# Heatmap (accepts numpy or tensor)
mat = np.random.rand(10, 10)
plotkit.plot_heatmap(mat, cmap='magma', annot=True)
plt.show()

# Line plot (accepts tensor input)
plotkit.plot_line(x, y, label='line')
plt.legend()
plt.show()

# Bar plot
plotkit.plot_bar(['A', 'B', 'C'], [1, 2, 3])
plt.show()

# Scatter plot
plotkit.plot_scatter(x, y + np.random.randn(100) * 0.1)
plt.show()
```

## License
MIT 