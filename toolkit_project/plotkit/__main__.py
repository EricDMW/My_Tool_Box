import argparse
import numpy as np
import matplotlib.pyplot as plt
from . import (
    plot_shadow_curve, plot_heatmap, plot_gray_scale, 
    plot_line, plot_bar, plot_scatter, set_research_style
)

def main():
    parser = argparse.ArgumentParser(description='plotkit demo CLI')
    parser.add_argument('--demo', type=str, default='shadow', 
                       choices=['shadow', 'heatmap', 'grayscale', 'line', 'bar', 'scatter', 'all'], 
                       help='Type of plot to demo')
    parser.add_argument('--style', type=str, default='research', 
                       choices=['research', 'default'], help='Plot style')
    args = parser.parse_args()

    # Set research style if requested
    if args.style == 'research':
        set_research_style()

    if args.demo == 'shadow':
        # Shadow curve demo
        x = np.arange(100)
        y1 = np.sin(x / 10) + np.random.normal(0, 0.1, (5, 100))
        y2 = np.cos(x / 10) + np.random.normal(0, 0.1, (5, 100))
        
        plot_shadow_curve([y1, y2], x=x, 
                         labels=['Sine', 'Cosine'],
                         title='Shadow Curve Demo',
                         xlabel='Time', ylabel='Amplitude',
                         style=args.style)
        
    elif args.demo == 'heatmap':
        # Heatmap demo
        data = np.random.rand(10, 10)
        xlabels = [f'F{i}' for i in range(10)]
        ylabels = [f'S{i}' for i in range(10)]
        
        plot_heatmap(data, xlabels=xlabels, ylabels=ylabels,
                    title='Heatmap Demo', annot=True,
                    style=args.style)
        
    elif args.demo == 'grayscale':
        # Grayscale demo
        data = np.random.rand(8, 8)
        plot_gray_scale(data, title='Grayscale Demo',
                       xlabel='X-axis', ylabel='Y-axis',
                       style=args.style)
        
    elif args.demo == 'line':
        # Line plot demo
        x = np.linspace(0, 2*np.pi, 100)
        y1, y2 = np.sin(x), np.cos(x)
        
        plot_line(x, [y1, y2], labels=['sin', 'cos'],
                 title='Line Plot Demo',
                 xlabel='Angle', ylabel='Value',
                 style=args.style)
        
    elif args.demo == 'bar':
        # Bar plot demo
        categories = ['A', 'B', 'C', 'D']
        values1 = [10, 20, 15, 25]
        values2 = [15, 18, 22, 19]
        
        plot_bar(categories, [values1, values2],
                labels=['Group 1', 'Group 2'],
                title='Bar Plot Demo',
                xlabel='Categories', ylabel='Values',
                style=args.style)
        
    elif args.demo == 'scatter':
        # Scatter plot demo
        x1, y1 = np.random.randn(50), np.random.randn(50)
        x2, y2 = np.random.randn(50), np.random.randn(50)
        
        plot_scatter([x1, x2], [y1, y2],
                    labels=['Group A', 'Group B'],
                    title='Scatter Plot Demo',
                    xlabel='X', ylabel='Y',
                    style=args.style)
        
    elif args.demo == 'all':
        # Show all plot types
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Shadow curve
        x = np.arange(50)
        y = np.random.randn(5, 50)
        plot_shadow_curve(y, x=x, ax=axes[0, 0], 
                         title='Shadow Curve', style=args.style)
        
        # Heatmap
        data = np.random.rand(6, 6)
        plot_heatmap(data, ax=axes[0, 1], title='Heatmap', style=args.style)
        
        # Grayscale
        plot_gray_scale(data, ax=axes[0, 2], title='Grayscale', style=args.style)
        
        # Line plot
        x = np.linspace(0, 2*np.pi, 50)
        y1, y2 = np.sin(x), np.cos(x)
        plot_line(x, [y1, y2], ax=axes[1, 0], 
                 title='Line Plot', style=args.style)
        
        # Bar plot
        categories = ['A', 'B', 'C']
        values = [10, 20, 15]
        plot_bar(categories, values, ax=axes[1, 1], 
                title='Bar Plot', style=args.style)
        
        # Scatter plot
        x, y = np.random.randn(30), np.random.randn(30)
        plot_scatter(x, y, ax=axes[1, 2], 
                    title='Scatter Plot', style=args.style)
        
        plt.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    main() 