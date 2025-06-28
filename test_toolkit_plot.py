#!/usr/bin/env python3
"""
Comprehensive demo script for plotkit functionality.
Tests all plotting functions with various input formats and research-quality styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotkit

def test_shadow_curves():
    """Test shadow curve functionality with various input formats."""
    print("Testing Shadow Curves...")
    
    # Create sample data
    x = np.arange(100)
    
    # Test 1: Single 2D tensor (RL training data simulation)
    print("  - Single 2D tensor (RL training simulation)")
    rewards = np.random.randn(10, 100) * 0.1 + np.log(x + 1)  # 10 runs, 100 episodes
    plotkit.plot_shadow_curve(rewards, x=x,
                             title='RL Training Progress (Single 2D Tensor)',
                             xlabel='Episodes', ylabel='Reward',
                             labels=['Agent Performance'])
    plt.savefig('test_shadow_single_2d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Multiple curves with different formats
    print("  - Multiple curves with different formats")
    curve1 = np.random.randn(5, 50)  # 5 runs, 50 timestamps
    curve2 = np.random.randn(5, 50)
    curve3 = np.sin(np.linspace(0, 4*np.pi, 50)) + np.random.normal(0, 0.1, 50)  # 1D array
    
    plotkit.plot_shadow_curve([curve1, curve2, curve3], 
                             labels=['Method A', 'Method B', 'Baseline'],
                             colors=['blue', 'red', 'green'],
                             title='Multi-Method Comparison',
                             xlabel='Time Steps', ylabel='Performance')
    plt.savefig('test_shadow_multiple.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 3: With custom standard deviation
    print("  - Custom standard deviation")
    y = np.random.randn(20, 50)
    custom_std = np.random.rand(50) * 0.5
    plotkit.plot_shadow_curve(y, y_std=custom_std,
                             title='Custom Error Bars',
                             xlabel='Time', ylabel='Value')
    plt.savefig('test_shadow_custom_std.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_shadow_curve_labels():
    """Test shadow curve label customization features."""
    print("Testing Shadow Curve Label Customization...")
    
    # Create sample data
    x = np.arange(50)
    y1 = np.random.randn(5, 50) * 0.1 + np.log(x + 1)
    y2 = np.random.randn(5, 50) * 0.1 + np.log(x + 1) * 0.8
    
    # Test 1: Basic label customization
    print("  - Basic label customization")
    plotkit.plot_shadow_curve([y1, y2], x=x,
                             labels=['Deep Q-Network', 'Proximal Policy Optimization'],
                             title='Reinforcement Learning Algorithm Comparison',
                             xlabel='Training Episodes', 
                             ylabel='Cumulative Reward',
                             legend=True)
    plt.savefig('test_shadow_labels_basic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Custom tick labels
    print("  - Custom tick labels")
    # Create custom x-axis tick labels (every 10 episodes)
    x_tick_positions = np.arange(0, 50, 10)
    x_tick_labels = [f'Episode {i}' for i in x_tick_positions]
    
    # Create custom y-axis tick labels
    y_tick_positions = np.arange(0, 5, 1)
    y_tick_labels = [f'{i:.1f}k' for i in y_tick_positions]
    
    plotkit.plot_shadow_curve([y1, y2], x=x,
                             legend_labels=['DQN', 'PPO'],  # Using legend_labels parameter
                             title='RL Training with Custom Tick Labels',
                             xlabel='Training Progress', 
                             ylabel='Performance (k units)',
                             x_tick_labels=x_tick_labels,
                             y_tick_labels=y_tick_labels)
    plt.savefig('test_shadow_labels_ticks.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 3: Advanced label customization with multiple curves
    print("  - Advanced label customization")
    y3 = np.random.randn(5, 50) * 0.1 + np.log(x + 1) * 1.2
    
    plotkit.plot_shadow_curve([y1, y2, y3], x=x,
                             labels=['Baseline DQN', 'Improved DQN', 'State-of-the-Art'],
                             colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                             title='Deep Q-Network Variants Performance',
                             xlabel='Training Episodes (×100)', 
                             ylabel='Average Reward per Episode',
                             legend=True,
                             grid=True)
    plt.savefig('test_shadow_labels_advanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 4: Scientific notation and special characters
    print("  - Scientific notation and special characters")
    plotkit.plot_shadow_curve([y1, y2], x=x,
                             labels=['α-DQN', 'β-PPO'],
                             title='Algorithm Performance: α vs β',
                             xlabel='Training Steps (×10³)', 
                             ylabel='Reward (×10⁻³)',
                             legend=True)
    plt.savefig('test_shadow_labels_scientific.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_heatmaps():
    """Test heatmap functionality."""
    print("Testing Heatmaps...")
    
    # Test 1: Basic heatmap
    print("  - Basic heatmap")
    data = np.random.rand(10, 10)
    plotkit.plot_heatmap(data, title='Basic Heatmap')
    plt.savefig('test_heatmap_basic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Heatmap with labels and annotations
    print("  - Heatmap with labels and annotations")
    xlabels = [f'Feature {i}' for i in range(10)]
    ylabels = [f'Sample {i}' for i in range(10)]
    plotkit.plot_heatmap(data, xlabels=xlabels, ylabels=ylabels,
                        title='Feature Matrix', annot=True, cmap='viridis')
    plt.savefig('test_heatmap_annotated.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 3: Correlation matrix
    print("  - Correlation matrix")
    # Create correlated data
    np.random.seed(42)
    features = np.random.randn(100, 8)
    corr_matrix = np.corrcoef(features.T)
    feature_names = [f'F{i}' for i in range(8)]
    
    plotkit.plot_heatmap(corr_matrix, xlabels=feature_names, ylabels=feature_names,
                        title='Feature Correlation Matrix',
                        annot=True, cmap='RdBu_r', cbar_label='Correlation')
    plt.savefig('test_heatmap_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_grayscale():
    """Test grayscale plotting functionality."""
    print("Testing Grayscale Plots...")
    
    # Test 1: Basic grayscale
    print("  - Basic grayscale")
    data = np.random.rand(8, 8)
    plotkit.plot_gray_scale(data, title='Grayscale Matrix',
                           xlabel='X-axis', ylabel='Y-axis')
    plt.savefig('test_grayscale_basic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Grayscale with custom range
    print("  - Grayscale with custom range")
    # Create data with specific pattern
    x, y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    pattern = np.exp(-(x**2 + y**2) / 2)
    plotkit.plot_gray_scale(pattern, title='Gaussian Pattern (Grayscale)',
                           xlabel='X', ylabel='Y')
    plt.savefig('test_grayscale_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_line_plots():
    """Test line plot functionality."""
    print("Testing Line Plots...")
    
    # Test 1: Single line
    print("  - Single line")
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    plotkit.plot_line(x, y, title='Sine Wave',
                     xlabel='Angle (radians)', ylabel='Value')
    plt.savefig('test_line_single.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Multiple lines
    print("  - Multiple lines")
    y1, y2, y3 = np.sin(x), np.cos(x), np.sin(2*x)
    plotkit.plot_line(x, [y1, y2, y3], 
                     labels=['sin(x)', 'cos(x)', 'sin(2x)'],
                     title='Trigonometric Functions',
                     xlabel='Angle (radians)', ylabel='Value')
    plt.savefig('test_line_multiple.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_bar_plots():
    """Test bar plot functionality."""
    print("Testing Bar Plots...")
    
    # Test 1: Single bar plot
    print("  - Single bar plot")
    categories = ['Method A', 'Method B', 'Method C', 'Method D']
    values = [0.85, 0.92, 0.88, 0.95]
    plotkit.plot_bar(categories, values, title='Method Comparison',
                    xlabel='Methods', ylabel='Accuracy')
    plt.savefig('test_bar_single.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Multiple bar groups
    print("  - Multiple bar groups")
    values1 = [0.85, 0.92, 0.88, 0.95]
    values2 = [0.82, 0.89, 0.85, 0.93]
    values3 = [0.80, 0.87, 0.83, 0.91]
    
    plotkit.plot_bar(categories, [values1, values2, values3],
                    labels=['Dataset 1', 'Dataset 2', 'Dataset 3'],
                    title='Multi-Dataset Method Comparison',
                    xlabel='Methods', ylabel='Accuracy')
    plt.savefig('test_bar_multiple.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_scatter_plots():
    """Test scatter plot functionality."""
    print("Testing Scatter Plots...")
    
    # Test 1: Single scatter
    print("  - Single scatter")
    np.random.seed(42)
    x = np.random.randn(100)
    y = x + np.random.randn(100) * 0.3
    plotkit.plot_scatter(x, y, title='Correlated Data',
                        xlabel='X', ylabel='Y')
    plt.savefig('test_scatter_single.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Multiple scatter groups
    print("  - Multiple scatter groups")
    x1, y1 = np.random.randn(50), np.random.randn(50)
    x2, y2 = np.random.randn(50) + 2, np.random.randn(50) + 2
    x3, y3 = np.random.randn(50) - 2, np.random.randn(50) - 2
    
    plotkit.plot_scatter([x1, x2, x3], [y1, y2, y3],
                        labels=['Group A', 'Group B', 'Group C'],
                        title='Multi-Group Scatter Plot',
                        xlabel='X', ylabel='Y')
    plt.savefig('test_scatter_multiple.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_research_styling():
    """Test research-quality styling."""
    print("Testing Research Styling...")
    
    # Set global research style
    plotkit.set_research_style()
    
    # Create a comprehensive example
    x = np.arange(50)
    y1 = np.random.randn(5, 50) * 0.1 + np.log(x + 1)
    y2 = np.random.randn(5, 50) * 0.1 + np.log(x + 1) * 0.8
    
    # Create subplots to show research styling
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Shadow curve
    plotkit.plot_shadow_curve([y1, y2], x=x, ax=axes[0, 0],
                             labels=['Method A', 'Method B'],
                             title='Learning Curves')
    
    # Heatmap
    data = np.random.rand(6, 6)
    plotkit.plot_heatmap(data, ax=axes[0, 1], title='Feature Matrix')
    
    # Line plot
    x_line = np.linspace(0, 2*np.pi, 50)
    y_line1, y_line2 = np.sin(x_line), np.cos(x_line)
    plotkit.plot_line(x_line, [y_line1, y_line2], ax=axes[1, 0],
                     labels=['sin', 'cos'], title='Trigonometric Functions')
    
    # Bar plot
    categories = ['A', 'B', 'C']
    values = [0.8, 0.9, 0.7]
    plotkit.plot_bar(categories, values, ax=axes[1, 1],
                    title='Performance Comparison')
    
    plt.tight_layout()
    plt.savefig('test_research_styling.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_tensor_support():
    """Test tensor input support (if available)."""
    print("Testing Tensor Support...")
    
    try:
        import torch
        print("  - PyTorch tensor support")
        
        # Create PyTorch tensors
        x_torch = torch.arange(50, dtype=torch.float32)
        y_torch = torch.randn(5, 50)
        
        plotkit.plot_shadow_curve(y_torch, x=x_torch,
                                 title='PyTorch Tensor Input',
                                 xlabel='Time', ylabel='Value')
        plt.savefig('test_tensor_pytorch.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("  - PyTorch not available, skipping tensor test")
    
    try:
        import tensorflow as tf
        print("  - TensorFlow tensor support")
        
        # Create TensorFlow tensors
        x_tf = tf.range(50, dtype=tf.float32)
        y_tf = tf.random.normal((5, 50))
        
        plotkit.plot_shadow_curve(y_tf, x=x_tf,
                                 title='TensorFlow Tensor Input',
                                 xlabel='Time', ylabel='Value')
        plt.savefig('test_tensor_tensorflow.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("  - TensorFlow not available, skipping tensor test")

def test_error_handling():
    """Test error handling capabilities."""
    print("Testing Error Handling...")
    
    # Test 1: None input
    print("  - None input handling")
    try:
        plotkit.plot_shadow_curve(None)
        print("    ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"    ✓ Correctly caught: {e}")
    
    # Test 2: Shape mismatch
    print("  - Shape mismatch handling")
    try:
        plotkit.plot_line([1, 2, 3], [1, 2])  # Different lengths
        print("    ERROR: Should have raised ValueError")
    except Exception as e:
        print(f"    ✓ Correctly caught: {e}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("PLOTKIT COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Run all tests
    test_shadow_curves()
    test_shadow_curve_labels()
    test_heatmaps()
    test_grayscale()
    test_line_plots()
    test_bar_plots()
    test_scatter_plots()
    test_research_styling()
    test_tensor_support()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("=" * 60)
    print("Generated test images:")
    print("- test_shadow_single_2d.png")
    print("- test_shadow_multiple.png")
    print("- test_shadow_custom_std.png")
    print("- test_shadow_labels_basic.png")
    print("- test_shadow_labels_ticks.png")
    print("- test_shadow_labels_advanced.png")
    print("- test_shadow_labels_scientific.png")
    print("- test_heatmap_basic.png")
    print("- test_heatmap_annotated.png")
    print("- test_heatmap_correlation.png")
    print("- test_grayscale_basic.png")
    print("- test_grayscale_pattern.png")
    print("- test_line_single.png")
    print("- test_line_multiple.png")
    print("- test_bar_single.png")
    print("- test_bar_multiple.png")
    print("- test_scatter_single.png")
    print("- test_scatter_multiple.png")
    print("- test_research_styling.png")
    print("- test_tensor_pytorch.png (if PyTorch available)")
    print("- test_tensor_tensorflow.png (if TensorFlow available)")
    print("\nAll images saved with research-quality styling!")

if __name__ == "__main__":
    main() 