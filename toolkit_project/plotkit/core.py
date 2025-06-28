import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _to_numpy(arr):
    """Convert input to numpy array, supporting torch and tensorflow tensors."""
    if hasattr(arr, 'detach') and callable(getattr(arr, 'detach', None)):
        # PyTorch tensor
        return arr.detach().cpu().numpy()
    elif hasattr(arr, 'numpy') and callable(getattr(arr, 'numpy', None)):
        # TensorFlow tensor or numpy array
        return arr.numpy()
    else:
        return np.asarray(arr)


def plot_shadow_curve(y, x=None, y_std=None, label=None, color=None, alpha=0.2, ax=None, axis=0, **kwargs):
    """
    Plot a line with a shadow (mean ± std or confidence interval), supporting numpy arrays, PyTorch, or TensorFlow tensors.
    Args:
        y: 1D or 2D array-like or tensor. If 2D, mean and std are computed along 'axis'.
        x: 1D array-like or tensor, optional. If None, uses np.arange(y.shape[axis])
        y_std: 1D array-like or tensor, optional. If None and y is 2D, uses std(y, axis=axis).
        label: str, label for the curve
        color: color for the line
        alpha: float, transparency for the shadow
        ax: matplotlib axis (optional)
        axis: int, axis to average over if y is 2D (default 0)
        **kwargs: passed to plt.plot
    """
    y = _to_numpy(y)
    if x is not None:
        x = _to_numpy(x)
    if y_std is not None:
        y_std = _to_numpy(y_std)
    ax = ax or plt.gca()

    # If y is 2D, compute mean and std along axis
    if y.ndim == 2:
        mean = np.mean(y, axis=axis)
        std = np.std(y, axis=axis) if y_std is None else y_std
        if x is None:
            x = np.arange(y.shape[1-axis])
    else:
        mean = y
        std = y_std
        if x is None:
            x = np.arange(len(y))

    line, = ax.plot(x, mean, label=label, color=color, **kwargs)
    if std is not None:
        std = _to_numpy(std)
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=alpha)
    return ax


def plot_heatmap(data, xlabels=None, ylabels=None, cmap='viridis', annot=False, ax=None, **kwargs):
    """
    Plot a heatmap.
    Args:
        data: 2D array-like
        xlabels: list of str, x-axis labels
        ylabels: list of str, y-axis labels
        cmap: str, colormap
        annot: bool, annotate cells
        ax: matplotlib axis (optional)
        **kwargs: passed to sns.heatmap
    """
    ax = ax or plt.gca()
    heatmap_kwargs = dict(cmap=cmap, annot=annot, ax=ax, **kwargs)
    if xlabels is not None:
        heatmap_kwargs['xticklabels'] = xlabels
    if ylabels is not None:
        heatmap_kwargs['yticklabels'] = ylabels
    sns.heatmap(_to_numpy(data), **heatmap_kwargs)
    return ax


def plot_line(x, y, label=None, color=None, ax=None, **kwargs):
    """
    Simple line plot.
    Args:
        x: 1D array-like
        y: 1D array-like
        label: str
        color: color
        ax: matplotlib axis (optional)
        **kwargs: passed to plt.plot
    """
    ax = ax or plt.gca()
    ax.plot(_to_numpy(x), _to_numpy(y), label=label, color=color, **kwargs)
    return ax


def plot_bar(x, height, label=None, color=None, ax=None, **kwargs):
    """
    Bar plot.
    Args:
        x: 1D array-like (categories or positions)
        height: 1D array-like (bar heights)
        label: str
        color: color
        ax: matplotlib axis (optional)
        **kwargs: passed to plt.bar
    """
    ax = ax or plt.gca()
    ax.bar(_to_numpy(x), _to_numpy(height), label=label, color=color, **kwargs)
    return ax


def plot_scatter(x, y, label=None, color=None, ax=None, **kwargs):
    """
    Scatter plot.
    Args:
        x: 1D array-like
        y: 1D array-like
        label: str
        color: color
        ax: matplotlib axis (optional)
        **kwargs: passed to plt.scatter
    """
    ax = ax or plt.gca()
    ax.scatter(_to_numpy(x), _to_numpy(y), label=label, color=color, **kwargs)
    return ax 