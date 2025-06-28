import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Union, List, Optional, Tuple, Any
import warnings

# Research-quality color palette (Nature, Science style)
RESEARCH_COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e', 
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf'
}

RESEARCH_COLOR_LIST = list(RESEARCH_COLORS.values())

def _to_numpy(arr):
    """Convert input to numpy array, supporting torch and tensorflow tensors."""
    if arr is None:
        return None
    
    if hasattr(arr, 'detach') and callable(getattr(arr, 'detach', None)):
        # PyTorch tensor
        return arr.detach().cpu().numpy()
    elif hasattr(arr, 'numpy') and callable(getattr(arr, 'numpy', None)):
        # TensorFlow tensor or numpy array
        return arr.numpy()
    else:
        return np.asarray(arr)

def _normalize_input(y_input):
    """
    Normalize input to handle various formats:
    - Single tensor/array: (m, n) -> [(m, n)]
    - List of tensors/arrays: [tensor1, tensor2, ...] -> [tensor1, tensor2, ...]
    - Multiple arguments: (y1, y2, ...) -> [y1, y2, ...]
    """
    if isinstance(y_input, (list, tuple)):
        # Handle list/tuple of inputs
        return [_to_numpy(y) for y in y_input]
    else:
        # Handle single input
        return [_to_numpy(y_input)]

def _get_color(color, index=0):
    """Get color from various input formats."""
    if color is None:
        return RESEARCH_COLOR_LIST[index % len(RESEARCH_COLOR_LIST)]
    elif isinstance(color, (list, tuple)):
        return color[index % len(color)]
    else:
        return color

def _get_label(label, index=0):
    """Get label from various input formats."""
    if label is None:
        return f'Curve {index + 1}'
    elif isinstance(label, (list, tuple)):
        return label[index % len(label)]
    else:
        return label

def plot_shadow_curve(y, x=None, y_std=None, labels=None, colors=None, alpha=0.2, 
                     ax=None, axis=0, figsize=(10, 6), title=None, xlabel=None, 
                     ylabel=None, legend=True, grid=True, style='research', 
                     legend_labels=None, x_tick_labels=None, y_tick_labels=None,
                     **kwargs):
    """
    Plot shadow curves with research-quality styling.
    
    Args:
        y: Input data in various formats:
           - Single tensor/array: (m, n) where n is timestamps, m is samples
           - List of tensors/arrays: [tensor1, tensor2, ...]
           - Multiple arguments: y1, y2, ...
        x: x-axis values (optional, auto-generated if None)
        y_std: Standard deviation values (optional, computed if None)
        labels: Labels for curves in legend (str, list, or None for auto)
        colors: Colors for curves (str, list, or None for auto)
        alpha: Transparency for shadows
        ax: Matplotlib axis (optional)
        axis: Axis to average over for 2D inputs (default 0)
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend: Whether to show legend
        grid: Whether to show grid
        style: Plot style ('research', 'default')
        legend_labels: Alternative to 'labels' parameter (for clarity)
        x_tick_labels: Custom x-axis tick labels
        y_tick_labels: Custom y-axis tick labels
        **kwargs: Additional arguments passed to plt.plot
    """
    # Use legend_labels if provided, otherwise use labels
    if legend_labels is not None:
        labels = legend_labels
    
    # Normalize input to list of arrays
    y_list = _normalize_input(y)
    
    # Set up plotting style
    if style == 'research':
        plt.style.use('default')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Process each curve
    for i, y_data in enumerate(y_list):
        if y_data is None:
            continue
            
        # Handle 2D input: compute mean and std
        if y_data.ndim == 2:
            mean = np.mean(y_data, axis=axis)
            std = np.std(y_data, axis=axis) if y_std is None else _to_numpy(y_std)
            if x is None:
                x_data = np.arange(y_data.shape[1-axis])
            else:
                x_data = _to_numpy(x)
        else:
            mean = y_data
            std = _to_numpy(y_std)
            if x is None:
                x_data = np.arange(len(y_data))
            else:
                x_data = _to_numpy(x)
        
        # Ensure x_data is not None
        if x_data is None:
            continue
        
        # Get color and label
        color = _get_color(colors, i)
        label = _get_label(labels, i)
        
        # Plot line
        line, = ax.plot(x_data, mean, label=label, color=color, linewidth=2, **kwargs)
        
        # Plot shadow
        if std is not None:
            ax.fill_between(x_data, mean - std, mean + std, 
                           color=color, alpha=alpha, linewidth=0)
    
    # Styling
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    
    # Custom tick labels
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
    if y_tick_labels is not None:
        ax.set_yticklabels(y_tick_labels)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    if legend and len(y_list) > 1:
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    return ax

def plot_heatmap(data, xlabels=None, ylabels=None, cmap='viridis', annot=False, 
                ax=None, figsize=(8, 6), title=None, xlabel=None, ylabel=None, 
                cbar=True, cbar_label=None, style='research', **kwargs):
    """
    Plot heatmap with research-quality styling.
    
    Args:
        data: 2D array-like data
        xlabels: X-axis labels
        ylabels: Y-axis labels
        cmap: Colormap name
        annot: Whether to annotate cells
        ax: Matplotlib axis (optional)
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cbar: Whether to show colorbar
        cbar_label: Colorbar label
        style: Plot style ('research', 'default')
        **kwargs: Additional arguments passed to sns.heatmap
    """
    data = _to_numpy(data)
    
    if data is None:
        raise ValueError("Data cannot be None")
    
    if style == 'research':
        plt.style.use('default')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare heatmap arguments
    heatmap_kwargs = {
        'cmap': cmap,
        'annot': annot,
        'ax': ax,
        'cbar': cbar,
        'square': False,
        'linewidths': 0.5,
        'linecolor': 'white',
        **kwargs
    }
    
    if xlabels is not None:
        heatmap_kwargs['xticklabels'] = xlabels
    if ylabels is not None:
        heatmap_kwargs['yticklabels'] = ylabels
    
    # Create heatmap
    sns.heatmap(data, **heatmap_kwargs)
    
    # Styling
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    
    # Colorbar styling
    if cbar and cbar_label:
        cbar_ax = ax.figure.axes[-1]
        cbar_ax.set_ylabel(cbar_label, fontsize=12)
    
    return ax

def plot_gray_scale(data, xlabels=None, ylabels=None, ax=None, figsize=(8, 6), 
                   title=None, xlabel=None, ylabel=None, style='research', **kwargs):
    """
    Plot grayscale heatmap (specialized for grayscale data).
    
    Args:
        data: 2D array-like data
        xlabels: X-axis labels
        ylabels: Y-axis labels
        ax: Matplotlib axis (optional)
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        style: Plot style ('research', 'default')
        **kwargs: Additional arguments
    """
    return plot_heatmap(data, xlabels=xlabels, ylabels=ylabels, cmap='gray', 
                       ax=ax, figsize=figsize, title=title, xlabel=xlabel, 
                       ylabel=ylabel, style=style, **kwargs)

def plot_line(x, y, labels=None, colors=None, ax=None, figsize=(10, 6), 
              title=None, xlabel=None, ylabel=None, legend=True, grid=True, 
              style='research', **kwargs):
    """
    Simple line plot with research-quality styling.
    
    Args:
        x: X-axis data (single array or list of arrays)
        y: Y-axis data (single array or list of arrays)
        labels: Labels for lines
        colors: Colors for lines
        ax: Matplotlib axis (optional)
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend: Whether to show legend
        grid: Whether to show grid
        style: Plot style ('research', 'default')
        **kwargs: Additional arguments passed to plt.plot
    """
    # Normalize inputs
    x_list = _normalize_input(x)
    y_list = _normalize_input(y)
    
    if style == 'research':
        plt.style.use('default')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each line
    for i, (x_data, y_data) in enumerate(zip(x_list, y_list)):
        if x_data is None or y_data is None:
            continue
            
        color = _get_color(colors, i)
        label = _get_label(labels, i)
        
        ax.plot(x_data, y_data, label=label, color=color, linewidth=2, **kwargs)
    
    # Styling
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    if legend and len(y_list) > 1:
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    return ax

def plot_bar(x, height, labels=None, colors=None, ax=None, figsize=(10, 6), 
             title=None, xlabel=None, ylabel=None, legend=True, grid=True, 
             style='research', **kwargs):
    """
    Bar plot with research-quality styling.
    
    Args:
        x: X-axis categories
        height: Bar heights (single array or list of arrays)
        labels: Labels for bars
        colors: Colors for bars
        ax: Matplotlib axis (optional)
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend: Whether to show legend
        grid: Whether to show grid
        style: Plot style ('research', 'default')
        **kwargs: Additional arguments passed to plt.bar
    """
    height_list = _normalize_input(height)
    
    if style == 'research':
        plt.style.use('default')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    x_data = _to_numpy(x)
    
    if x_data is None:
        raise ValueError("X-axis data cannot be None")
    
    # Plot bars
    for i, h_data in enumerate(height_list):
        if h_data is None:
            continue
            
        color = _get_color(colors, i)
        label = _get_label(labels, i)
        
        ax.bar(x_data, h_data, label=label, color=color, alpha=0.8, **kwargs)
    
    # Styling
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    
    if legend and len(height_list) > 1:
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    return ax

def plot_scatter(x, y, labels=None, colors=None, ax=None, figsize=(10, 6), 
                title=None, xlabel=None, ylabel=None, legend=True, grid=True, 
                style='research', **kwargs):
    """
    Scatter plot with research-quality styling.
    
    Args:
        x: X-axis data (single array or list of arrays)
        y: Y-axis data (single array or list of arrays)
        labels: Labels for scatter points
        colors: Colors for scatter points
        ax: Matplotlib axis (optional)
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend: Whether to show legend
        grid: Whether to show grid
        style: Plot style ('research', 'default')
        **kwargs: Additional arguments passed to plt.scatter
    """
    # Normalize inputs
    x_list = _normalize_input(x)
    y_list = _normalize_input(y)
    
    if style == 'research':
        plt.style.use('default')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each scatter
    for i, (x_data, y_data) in enumerate(zip(x_list, y_list)):
        if x_data is None or y_data is None:
            continue
            
        color = _get_color(colors, i)
        label = _get_label(labels, i)
        
        ax.scatter(x_data, y_data, label=label, color=color, alpha=0.7, s=50, **kwargs)
    
    # Styling
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    if legend and len(y_list) > 1:
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    return ax

def set_research_style():
    """Set matplotlib to research journal quality styling."""
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1 