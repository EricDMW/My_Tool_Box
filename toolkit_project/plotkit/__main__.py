import argparse
import numpy as np
import matplotlib.pyplot as plt
from . import plot_shadow_curve, plot_heatmap, plot_line, plot_bar, plot_scatter

def main():
    parser = argparse.ArgumentParser(description='plotkit demo CLI')
    parser.add_argument('--demo', type=str, default='shadow', choices=['shadow', 'heatmap', 'line', 'bar', 'scatter'], help='Type of plot to demo')
    args = parser.parse_args()

    if args.demo == 'shadow':
        x = np.arange(100)
        y = np.sin(x / 10)
        y_std = 0.1 + 0.1 * np.abs(np.cos(x / 10))
        plot_shadow_curve(x, y, y_std, label='sin(x)', color='blue')
        plt.legend()
        plt.title('Shadow Curve Demo')
    elif args.demo == 'heatmap':
        mat = np.random.rand(10, 10)
        plot_heatmap(mat, cmap='magma', annot=True)
        plt.title('Heatmap Demo')
    elif args.demo == 'line':
        x = np.arange(100)
        y = np.sin(x / 10)
        plot_line(x, y, label='line')
        plt.legend()
        plt.title('Line Plot Demo')
    elif args.demo == 'bar':
        plot_bar(['A', 'B', 'C'], [1, 2, 3])
        plt.title('Bar Plot Demo')
    elif args.demo == 'scatter':
        x = np.arange(100)
        y = np.sin(x / 10) + np.random.randn(100) * 0.1
        plot_scatter(x, y)
        plt.title('Scatter Plot Demo')
    plt.show()

if __name__ == '__main__':
    main() 