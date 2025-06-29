#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a map from a .cfg/.yaml map file using matplotlib.
Usage: python plot_map.py --map obstacles05
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

from env import maps

def load_map(map_name):
    map_dir = os.path.dirname(maps.__file__)
    map_path = os.path.join(map_dir, map_name)
    cfg_path = map_path + '.cfg'
    yaml_path = map_path + '.yaml'
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        map_config = yaml.safe_load(f)
    mapdim = map_config['mapdim']
    mapmin = map_config['mapmin']
    mapmax = map_config['mapmax']
    if os.path.exists(cfg_path):
        try:
            map_data = np.loadtxt(cfg_path)
            if map_data.ndim == 1:
                map_data = map_data.reshape(mapdim)
            map_data = map_data.astype(float)
        except Exception as e:
            print(f"Warning: Could not load map data from {cfg_path}: {e}")
            map_data = np.zeros(mapdim, dtype=float)
    else:
        map_data = np.zeros(mapdim, dtype=float)
    return map_data, mapmin, mapmax

def plot_map(map_data, mapmin, mapmax, map_name):
    plt.figure(figsize=(8,8))
    plt.imshow(map_data, cmap='gray_r', origin='lower',
               extent=[mapmin[0], mapmax[0], mapmin[1], mapmax[1]])
    plt.title(f"Map: {map_name}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.savefig('map_plot.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot a map from a .cfg/.yaml map file')
    parser.add_argument('--map', type=str, default='obstacles05', help='Map name (e.g., obstacles05)')
    args = parser.parse_args()
    map_data, mapmin, mapmax = load_map(args.map)
    plot_map(map_data, mapmin, mapmax, args.map)

if __name__ == '__main__':
    main() 