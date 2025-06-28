#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : build_map.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: RL_LLM
@Date   : 10/05/2024
@Time   : 18:56:07
@Info   : Simple Map Builder from YAML to CFG
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

def main(yaml_file_path):
    """
    Generate CFG file from YAML configuration.
    
    Args:
        yaml_file_path (str): Path to the YAML file containing map configuration
    """
    print(f"Building map from: {yaml_file_path}")
    
    # Load YAML configuration
    with open(yaml_file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract map information
    map_info = config['map_info']
    width_m = map_info['width']
    height_m = map_info['height']
    resolution = map_info['resolution']
    
    # Calculate grid dimensions
    grid_width = int(width_m / resolution)
    grid_height = int(height_m / resolution)
    
    # Create empty map
    map_data = np.zeros((grid_width, grid_height), dtype=np.int8)
    
    # Add boundary if enabled
    boundary = map_info.get('boundary', {})
    if boundary.get('enabled', True):
        add_boundary(map_data, boundary, resolution)
    
    # Add obstacles
    obstacles = config.get('obstacles', [])
    for obstacle in obstacles:
        add_obstacle(map_data, obstacle, resolution)
    
    # Save CFG file
    yaml_path = Path(yaml_file_path)
    cfg_path = yaml_path.with_suffix('.cfg')
    
    np.savetxt(cfg_path, map_data, fmt='%d')
    print(f"CFG file saved: {cfg_path}")
    
    # Print statistics
    total_cells = map_data.size
    occupied_cells = np.sum(map_data)
    print(f"Map size: {grid_width}x{grid_height} cells")
    print(f"Occupied cells: {occupied_cells}/{total_cells} ({occupied_cells/total_cells*100:.1f}%)")

def add_boundary(map_data, boundary_config, resolution):
    """Add boundary to the map."""
    boundary_style = boundary_config.get('style', 'rectangle')
    thickness = boundary_config.get('thickness', resolution)
    thickness_cells = max(1, int(thickness / resolution))
    
    if boundary_style == 'rectangle':
        # Simple rectangular boundary
        map_data[0:thickness_cells, :] = 1  # Bottom
        map_data[:, 0:thickness_cells] = 1  # Left
        map_data[-thickness_cells:, :] = 1  # Top
        map_data[:, -thickness_cells:] = 1  # Right

def add_obstacle(map_data, obstacle, resolution):
    """Add obstacle to the map."""
    obstacle_type = obstacle['type']
    
    if obstacle_type == 'rectangle':
        add_rectangle(map_data, obstacle, resolution)
    elif obstacle_type == 'circle':
        add_circle(map_data, obstacle, resolution)
    elif obstacle_type == 'triangle':
        add_triangle(map_data, obstacle, resolution)
    elif obstacle_type == 'hexagon':
        add_hexagon(map_data, obstacle, resolution)
    elif obstacle_type == 'polygon':
        add_polygon(map_data, obstacle, resolution)
    elif obstacle_type == 'line':
        add_line(map_data, obstacle, resolution)

def add_rectangle(map_data, obstacle, resolution):
    """Add rectangular obstacle."""
    center = obstacle['center']
    width = obstacle['width']
    height = obstacle['height']
    angle = obstacle.get('angle', 0.0)
    filled = obstacle.get('filled', True)
    
    center_x = int(center[0] / resolution)
    center_y = int(center[1] / resolution)
    width_cells = int(width / resolution)
    height_cells = int(height / resolution)
    
    half_width = width_cells // 2
    half_height = height_cells // 2
    
    if abs(angle) < 0.01:  # No rotation
        x1 = max(0, center_x - half_width)
        x2 = min(map_data.shape[0], center_x + half_width)
        y1 = max(0, center_y - half_height)
        y2 = min(map_data.shape[1], center_y + half_height)
        
        if filled:
            map_data[x1:x2, y1:y2] = 1
        else:
            # Hollow rectangle
            map_data[x1:x2, y1:y1+1] = 1
            map_data[x1:x2, y2-1:y2] = 1
            map_data[x1:x1+1, y1:y2] = 1
            map_data[x2-1:x2, y1:y2] = 1

def add_circle(map_data, obstacle, resolution):
    """Add circular obstacle."""
    center = obstacle['center']
    radius = obstacle['radius']
    filled = obstacle.get('filled', True)
    
    center_x = int(center[0] / resolution)
    center_y = int(center[1] / resolution)
    radius_cells = int(radius / resolution)
    
    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            distance = math.sqrt((i - center_x)**2 + (j - center_y)**2)
            if filled:
                if distance <= radius_cells:
                    map_data[i, j] = 1
            else:
                if distance <= radius_cells and distance > radius_cells - 1:
                    map_data[i, j] = 1

def add_triangle(map_data, obstacle, resolution):
    """Add triangular obstacle."""
    vertices = obstacle['vertices']
    filled = obstacle.get('filled', True)
    
    grid_vertices = []
    for vertex in vertices:
        x = int(vertex[0] / resolution)
        y = int(vertex[1] / resolution)
        grid_vertices.append([x, y])
    
    if len(grid_vertices) == 3:
        if filled:
            fill_triangle(map_data, grid_vertices)
        else:
            # Draw triangle outline
            for i in range(3):
                start = grid_vertices[i]
                end = grid_vertices[(i + 1) % 3]
                draw_line(map_data, start, end)

def add_hexagon(map_data, obstacle, resolution):
    """Add hexagonal obstacle."""
    center = obstacle['center']
    radius = obstacle['radius']
    angle = obstacle.get('angle', 0.0)
    filled = obstacle.get('filled', True)
    
    center_x = int(center[0] / resolution)
    center_y = int(center[1] / resolution)
    radius_cells = int(radius / resolution)
    
    # Generate hexagon vertices
    vertices = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = center_x + radius_cells * math.cos(theta)
        y = center_y + radius_cells * math.sin(theta)
        vertices.append([int(x), int(y)])
    
    if filled:
        fill_polygon(map_data, vertices)
    else:
        # Draw hexagon outline
        for i in range(6):
            start = vertices[i]
            end = vertices[(i + 1) % 6]
            draw_line(map_data, start, end)

def add_polygon(map_data, obstacle, resolution):
    """Add polygon obstacle."""
    points = obstacle['points']
    filled = obstacle.get('filled', True)
    
    grid_points = []
    for point in points:
        x = int(point[0] / resolution)
        y = int(point[1] / resolution)
        grid_points.append([x, y])
    
    if len(grid_points) >= 3:
        if filled:
            fill_polygon(map_data, grid_points)
        else:
            # Draw polygon outline
            for i in range(len(grid_points)):
                start = grid_points[i]
                end = grid_points[(i + 1) % len(grid_points)]
                draw_line(map_data, start, end)

def add_line(map_data, obstacle, resolution):
    """Add line obstacle."""
    start = obstacle['start']
    end = obstacle['end']
    width = obstacle.get('width', resolution)
    
    start_x = int(start[0] / resolution)
    start_y = int(start[1] / resolution)
    end_x = int(end[0] / resolution)
    end_y = int(end[1] / resolution)
    width_cells = max(1, int(width / resolution))
    
    draw_line(map_data, [start_x, start_y], [end_x, end_y], width_cells)

def fill_triangle(map_data, vertices):
    """Fill a triangle using barycentric coordinates."""
    if len(vertices) != 3:
        return
    
    v0, v1, v2 = vertices
    
    # Find bounding box
    min_x = max(0, min(v[0] for v in vertices))
    max_x = min(map_data.shape[0] - 1, max(v[0] for v in vertices))
    min_y = max(0, min(v[1] for v in vertices))
    max_y = min(map_data.shape[1] - 1, max(v[1] for v in vertices))
    
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if point_in_triangle(x, y, v0, v1, v2):
                map_data[x, y] = 1

def point_in_triangle(x, y, v0, v1, v2):
    """Check if point is inside triangle using barycentric coordinates."""
    denominator = ((v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
    if denominator == 0:
        return False
    
    a = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denominator
    b = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denominator
    c = 1.0 - a - b
    
    return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1

def fill_polygon(map_data, vertices):
    """Fill a polygon using scan line algorithm."""
    if len(vertices) < 3:
        return
    
    # Find bounding box
    min_x = max(0, min(v[0] for v in vertices))
    max_x = min(map_data.shape[0] - 1, max(v[0] for v in vertices))
    min_y = max(0, min(v[1] for v in vertices))
    max_y = min(map_data.shape[1] - 1, max(v[1] for v in vertices))
    
    # Scan line algorithm
    for y in range(min_y, max_y + 1):
        intersections = []
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            
            if (v1[1] <= y and v2[1] > y) or (v2[1] <= y and v1[1] > y):
                if v2[1] != v1[1]:
                    x = v1[0] + (y - v1[1]) * (v2[0] - v1[0]) / (v2[1] - v1[1])
                    intersections.append(int(x))
        
        intersections.sort()
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x1 = max(0, intersections[i])
                x2 = min(map_data.shape[0] - 1, intersections[i + 1])
                map_data[x1:x2 + 1, y] = 1

def draw_line(map_data, start, end, width=1):
    """Draw a line between two points."""
    x0, y0 = start
    x1, y1 = end
    
    # Bresenham's line algorithm
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        # Add width around the point
        for dx in range(-width//2, width//2 + 1):
            for dy in range(-width//2, width//2 + 1):
                nx, ny = x0 + dx, y0 + dy
                if 0 <= nx < map_data.shape[0] and 0 <= ny < map_data.shape[1]:
                    map_data[nx, ny] = 1
        
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

if __name__ == "__main__":
    # Get YAML file path from command line argument
    
    
    yaml_file_path = "/home/dongmingwang/project/My_Tool_Box/envlib_project/env_lib/ajlatt_env/env/maps/sample_map_config.yaml"
    main(yaml_file_path) 