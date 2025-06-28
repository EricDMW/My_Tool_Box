# plotkit

A comprehensive, research-quality plotting toolkit for reinforcement learning and research applications.

## Features

- **Research-quality styling**: Publication-ready plots with journal standards
- **Flexible input formats**: Support for numpy arrays, PyTorch tensors, TensorFlow tensors, and lists
- **Advanced shadow curves**: Handle multiple curves, 2D tensors, and automatic statistics
- **Enhanced heatmaps**: Professional heatmaps with customizable styling
- **Grayscale plotting**: Specialized grayscale heatmaps for scientific visualization
- **Multi-curve support**: Plot multiple datasets with automatic color cycling
- **Robust error handling**: Comprehensive input validation and error messages

## Installation

```bash
# From the toolkit_project directory
pip install -e .
```

## Quick Start

```python
import numpy as np
import plotkit
import matplotlib.pyplot as plt

# Set research-quality styling globally
plotkit.set_research_style()

# Create sample data
x = np.arange(100)
y1 = np.sin(x / 10) + np.random.normal(0, 0.1, 100)
y2 = np.cos(x / 10) + np.random.normal(0, 0.1, 100)

# Plot shadow curves
plotkit.plot_shadow_curve([y1, y2], labels=['Sine', 'Cosine'], 
                         title='Oscillating Functions', xlabel='Time', ylabel='Amplitude')
plt.show()
```

## Core Functions

### Shadow Curves (`plot_shadow_curve`)

Plot curves with confidence intervals or standard deviations.

```python
# Single 2D tensor (m samples, n timestamps)
data_2d = np.random.randn(10, 100)  # 10 samples, 100 timestamps
plotkit.plot_shadow_curve(data_2d, title='Learning Curve')

# Multiple curves
curve1 = np.random.randn(5, 50)
curve2 = np.random.randn(5, 50)
plotkit.plot_shadow_curve([curve1, curve2], 
                         labels=['Method A', 'Method B'],
                         colors=['blue', 'red'])

# With custom x-axis and std
x = np.linspace(0, 10, 50)
y = np.random.randn(20, 50)
std = np.random.rand(50) * 0.5
plotkit.plot_shadow_curve(y, x=x, y_std=std, title='Custom Error Bars')

# Advanced label customization
plotkit.plot_shadow_curve([y1, y2], x=x,
                         labels=['Deep Q-Network', 'Proximal Policy Optimization'],
                         title='RL Algorithm Comparison',
                         xlabel='Training Episodes', 
                         ylabel='Cumulative Reward',
                         legend_labels=['DQN', 'PPO'],  # Alternative to labels
                         x_tick_labels=['Ep 0', 'Ep 10', 'Ep 20', 'Ep 30', 'Ep 40'],
                         y_tick_labels=['0k', '1k', '2k', '3k', '4k'])
```

### Heatmaps (`plot_heatmap`)

Create professional heatmaps with customizable styling.

```python
# Basic heatmap
data = np.random.rand(10, 10)
plotkit.plot_heatmap(data, title='Correlation Matrix')

# With labels and annotations
xlabels = [f'Feature {i}' for i in range(10)]
ylabels = [f'Sample {i}' for i in range(10)]
plotkit.plot_heatmap(data, xlabels=xlabels, ylabels=ylabels, 
                    annot=True, cmap='viridis', title='Feature Matrix')
```

### Grayscale Heatmaps (`plot_gray_scale`)

Specialized grayscale plotting for scientific visualization.

```python
# Grayscale heatmap
data = np.random.rand(8, 8)
plotkit.plot_gray_scale(data, title='Grayscale Matrix', 
                       xlabel='X-axis', ylabel='Y-axis')
```

### Line Plots (`plot_line`)

Simple line plots with research-quality styling.

```python
# Single line
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plotkit.plot_line(x, y, title='Sine Wave', xlabel='Angle', ylabel='Value')

# Multiple lines
y1, y2 = np.sin(x), np.cos(x)
plotkit.plot_line(x, [y1, y2], labels=['sin', 'cos'], 
                 title='Trigonometric Functions')
```

### Bar Plots (`plot_bar`)

Bar plots with professional styling.

```python
# Single bar plot
categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]
plotkit.plot_bar(categories, values, title='Bar Chart', 
                xlabel='Categories', ylabel='Values')

# Multiple bar groups
values1, values2 = [10, 20, 15, 25], [15, 18, 22, 19]
plotkit.plot_bar(categories, [values1, values2], 
                labels=['Group 1', 'Group 2'])
```

### Scatter Plots (`plot_scatter`)

Scatter plots with customizable styling.

```python
# Single scatter
x = np.random.randn(100)
y = np.random.randn(100)
plotkit.plot_scatter(x, y, title='Scatter Plot')

# Multiple scatter groups
x1, y1 = np.random.randn(50), np.random.randn(50)
x2, y2 = np.random.randn(50), np.random.randn(50)
plotkit.plot_scatter([x1, x2], [y1, y2], 
                    labels=['Group A', 'Group B'])
```

## Advanced Features

### Input Format Flexibility

```python
import torch
import tensorflow as tf

# PyTorch tensors
torch_data = torch.randn(10, 100)
plotkit.plot_shadow_curve(torch_data)

# TensorFlow tensors
tf_data = tf.random.normal((10, 100))
plotkit.plot_shadow_curve(tf_data)

# Mixed formats
data_list = [np.random.randn(5, 50), torch.randn(5, 50)]
plotkit.plot_shadow_curve(data_list)
```

### Research-Quality Styling

```python
# Set global research style
plotkit.set_research_style()

# Use research colors
colors = plotkit.RESEARCH_COLORS
plotkit.plot_line(x, y, color=colors['blue'])

# Custom styling per plot
plotkit.plot_shadow_curve(y, style='research', 
                         title='Publication-Ready Plot')
```

### Multi-Curve Support

```python
# Automatic color cycling
curves = [np.random.randn(5, 50) for _ in range(4)]
plotkit.plot_shadow_curve(curves, 
                         labels=['Method A', 'Method B', 'Method C', 'Method D'])

# Custom colors and labels
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
labels = ['Baseline', 'Improved', 'Best', 'Oracle']
plotkit.plot_shadow_curve(curves, colors=colors, labels=labels)
```

## Configuration

### Research Styling

```python
# Apply research styling globally
plotkit.set_research_style()

# Or per-plot
plotkit.plot_shadow_curve(y, style='research')
```

### Color Palettes

```python
# Access research colors
print(plotkit.RESEARCH_COLORS)
print(plotkit.RESEARCH_COLOR_LIST)

# Use custom colors
custom_colors = ['#ff0000', '#00ff00', '#0000ff']
plotkit.plot_shadow_curve(y, colors=custom_colors)
```

## Error Handling

The toolkit provides robust error handling:

```python
# Invalid data
try:
    plotkit.plot_shadow_curve(None)
except ValueError as e:
    print(f"Error: {e}")

# Shape mismatches
try:
    plotkit.plot_line([1, 2, 3], [1, 2])  # Different lengths
except ValueError as e:
    print(f"Error: {e}")
```

## Examples

### Reinforcement Learning Training Curves

```python
# Simulate RL training data
episodes = np.arange(1000)
rewards = np.random.randn(10, 1000) * 0.1 + np.log(episodes + 1)

plotkit.plot_shadow_curve(rewards, x=episodes,
                         title='RL Training Progress',
                         xlabel='Episodes', ylabel='Reward',
                         labels=['Agent Performance'])
plt.show()
```

### Confusion Matrix

```python
# Create confusion matrix
confusion = np.array([[85, 15], [10, 90]])
labels = ['Predicted Negative', 'Predicted Positive']

plotkit.plot_heatmap(confusion, 
                    xlabels=labels, ylabels=labels,
                    title='Confusion Matrix',
                    annot=True, cmap='Blues')
plt.show()
```

### Multi-Method Comparison

```python
# Compare multiple methods
methods = ['Method A', 'Method B', 'Method C', 'Method D']
accuracies = [0.85, 0.92, 0.88, 0.95]
std_errors = [0.02, 0.01, 0.03, 0.01]

plotkit.plot_bar(methods, accuracies, 
                title='Method Comparison',
                xlabel='Methods', ylabel='Accuracy')
plt.show()
```

## Contributing

This toolkit is designed to be extensible. To add new plot types:

1. Add the function to `core.py`
2. Update `__init__.py` to export the function
3. Add documentation and examples
4. Include proper error handling and input validation

## License

MIT License - see LICENSE file for details. 