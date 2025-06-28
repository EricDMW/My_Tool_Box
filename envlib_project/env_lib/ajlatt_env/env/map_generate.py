#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : map_generate.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: RL_LLM
@Date   : 10/05/2024
@Time   : 18:56:07
@Info   : Advanced Map Generation System for AJLATT Environment
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import math

# Add the maps directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'maps'))
from maps.map_utils import generate_map as basic_generate_map

class AdvancedMapGenerator:
    """
    Advanced map generator that reads YAML configuration files and generates
    all necessary map files including obstacles, boundaries, and metadata.
    """
    
    def __init__(self, yaml_path: str):
        """
        Initialize the map generator with a YAML configuration file.
        
        Args:
            yaml_path: Path to the YAML configuration file
        """
        self.yaml_path = yaml_path
        self.config = self._load_yaml_config()
        self.map_data = None
        self.grid_width = None
        self.grid_height = None
        
    def _load_yaml_config(self) -> Dict:
        """Load and validate the YAML configuration file."""
        try:
            with open(self.yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Check if this is a legacy format or new comprehensive format
            if 'map_info' in config:
                # New comprehensive format
                required_fields = ['map_info']
                if 'obstacles' not in config:
                    config['obstacles'] = []
            else:
                # Legacy format - convert to new format
                config = self._convert_legacy_config(config)
            
            return config
        except Exception as e:
            raise ValueError(f"Error loading YAML file: {e}")
    
    def _convert_legacy_config(self, legacy_config: Dict) -> Dict:
        """Convert legacy YAML format to new comprehensive format."""
        # Extract map dimensions from legacy format
        mapdim = legacy_config.get('mapdim', [181, 181])
        mapmax = legacy_config.get('mapmax', [72.4, 72.4])
        mapres = legacy_config.get('mapres', [0.4, 0.4])
        
        # Create new format
        new_config = {
            'map_info': {
                'name': Path(self.yaml_path).stem,
                'width': mapmax[0],
                'height': mapmax[1],
                'resolution': mapres[0],
                'boundary': {
                    'enabled': True,
                    'thickness': mapres[0],
                    'style': 'rectangle',
                    'rectangle': {'margin': 0.0}
                }
            },
            'obstacles': []
        }
        
        return new_config
    
    def _calculate_grid_dimensions(self) -> Tuple[int, int]:
        """Calculate grid dimensions based on map info."""
        map_info = self.config['map_info']
        width_m = map_info['width']
        height_m = map_info['height']
        resolution = map_info['resolution']
        
        self.grid_width = int(width_m / resolution)
        self.grid_height = int(height_m / resolution)
        
        return self.grid_width, self.grid_height
    
    def _create_empty_map(self) -> np.ndarray:
        """Create an empty map with boundary walls."""
        width, height = self._calculate_grid_dimensions()
        map_data = np.zeros((width, height), dtype=np.int8)
        
        # Add boundary based on configuration
        boundary_config = self.config['map_info'].get('boundary', {})
        if boundary_config.get('enabled', True):
            self._add_boundary(map_data, boundary_config)
        
        return map_data
    
    def _add_boundary(self, map_data: np.ndarray, boundary_config: Dict) -> np.ndarray:
        """Add boundary to the map based on configuration."""
        boundary_style = boundary_config.get('style', 'rectangle')
        thickness = boundary_config.get('thickness', self.config['map_info']['resolution'])
        thickness_cells = max(1, int(thickness / self.config['map_info']['resolution']))
        
        if boundary_style == 'rectangle':
            margin = boundary_config.get('rectangle', {}).get('margin', 0.0)
            margin_cells = int(margin / self.config['map_info']['resolution'])
            
            # Add rectangular boundary
            map_data[margin_cells:margin_cells+thickness_cells, margin_cells:self.grid_height-margin_cells] = 1.0  # Bottom
            map_data[margin_cells:self.grid_width-margin_cells, margin_cells:margin_cells+thickness_cells] = 1.0  # Left
            map_data[self.grid_width-margin_cells-thickness_cells:self.grid_width-margin_cells, margin_cells:self.grid_height-margin_cells] = 1.0  # Top
            map_data[margin_cells:self.grid_width-margin_cells, self.grid_height-margin_cells-thickness_cells:self.grid_height-margin_cells] = 1.0  # Right
            
        elif boundary_style == 'circle':
            circle_config = boundary_config.get('circle', {})
            center = circle_config.get('center', [self.config['map_info']['width']/2, self.config['map_info']['height']/2])
            radius = circle_config.get('radius', min(self.config['map_info']['width'], self.config['map_info']['height'])/2)
            
            center_x = int(center[0] / self.config['map_info']['resolution'])
            center_y = int(center[1] / self.config['map_info']['resolution'])
            radius_cells = int(radius / self.config['map_info']['resolution'])
            
            # Create circular boundary
            for i in range(self.grid_width):
                for j in range(self.grid_height):
                    distance = math.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if distance > radius_cells - thickness_cells and distance <= radius_cells:
                        map_data[i, j] = 1.0
                        
        elif boundary_style == 'polygon':
            polygon_config = boundary_config.get('polygon', {})
            points = polygon_config.get('points', [[0, 0], [self.config['map_info']['width'], 0], 
                                                  [self.config['map_info']['width'], self.config['map_info']['height']], [0, self.config['map_info']['height']]])
            
            # Convert points to grid coordinates
            grid_points = []
            for point in points:
                x = int(point[0] / self.config['map_info']['resolution'])
                y = int(point[1] / self.config['map_info']['resolution'])
                grid_points.append([x, y])
            
            # Create polygon boundary (simplified - just fill the polygon)
            if len(grid_points) >= 3:
                self._fill_polygon(map_data, grid_points)
        
        return map_data
    
    def _add_rectangular_obstacle(self, map_data: np.ndarray, obstacle: Dict) -> np.ndarray:
        """Add a rectangular obstacle to the map."""
        resolution = self.config['map_info']['resolution']
        
        # Extract obstacle parameters
        center = obstacle['center']
        width = obstacle['width']
        height = obstacle['height']
        angle = obstacle.get('angle', 0.0)  # Rotation angle in radians
        filled = obstacle.get('filled', True)
        
        # Convert to grid coordinates
        center_x = int(center[0] / resolution)
        center_y = int(center[1] / resolution)
        width_cells = int(width / resolution)
        height_cells = int(height / resolution)
        
        # Calculate corner positions
        half_width = width_cells // 2
        half_height = height_cells // 2
        
        # Create rotated rectangle
        if abs(angle) < 0.01:  # No rotation
            x1 = max(0, center_x - half_width)
            x2 = min(self.grid_width, center_x + half_width)
            y1 = max(0, center_y - half_height)
            y2 = min(self.grid_height, center_y + half_height)
            
            if filled:
                map_data[x1:x2, y1:y2] = 1.0
            else:
                # Create hollow rectangle
                map_data[x1:x2, y1:y1+1] = 1.0  # Bottom edge
                map_data[x1:x2, y2-1:y2] = 1.0  # Top edge
                map_data[x1:x1+1, y1:y2] = 1.0  # Left edge
                map_data[x2-1:x2, y1:y2] = 1.0  # Right edge
        else:
            # For rotated rectangles, we'll approximate with a larger rectangle
            # This is a simplified approach - for precise rotation, more complex algorithms are needed
            max_dim = max(width_cells, height_cells)
            x1 = max(0, center_x - max_dim)
            x2 = min(self.grid_width, center_x + max_dim)
            y1 = max(0, center_y - max_dim)
            y2 = min(self.grid_height, center_y + max_dim)
            map_data[x1:x2, y1:y2] = 1.0
        
        return map_data
    
    def _add_circular_obstacle(self, map_data: np.ndarray, obstacle: Dict) -> np.ndarray:
        """Add a circular obstacle to the map."""
        resolution = self.config['map_info']['resolution']
        
        center = obstacle['center']
        radius = obstacle['radius']
        filled = obstacle.get('filled', True)
        
        # Convert to grid coordinates
        center_x = int(center[0] / resolution)
        center_y = int(center[1] / resolution)
        radius_cells = int(radius / resolution)
        
        # Create circular obstacle
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                distance = math.sqrt((i - center_x)**2 + (j - center_y)**2)
                if filled:
                    if distance <= radius_cells:
                        map_data[i, j] = 1.0
                else:
                    # Create hollow circle
                    if distance <= radius_cells and distance > radius_cells - 1:
                        map_data[i, j] = 1.0
        
        return map_data
    
    def _add_triangular_obstacle(self, map_data: np.ndarray, obstacle: Dict) -> np.ndarray:
        """Add a triangular obstacle to the map."""
        resolution = self.config['map_info']['resolution']
        vertices = obstacle['vertices']
        filled = obstacle.get('filled', True)
        
        # Convert vertices to grid coordinates
        grid_vertices = []
        for vertex in vertices:
            x = int(vertex[0] / resolution)
            y = int(vertex[1] / resolution)
            grid_vertices.append([x, y])
        
        if len(grid_vertices) == 3:
            if filled:
                self._fill_triangle(map_data, grid_vertices)
            else:
                # Create triangle outline
                self._draw_triangle_outline(map_data, grid_vertices)
        
        return map_data
    
    def _add_hexagonal_obstacle(self, map_data: np.ndarray, obstacle: Dict) -> np.ndarray:
        """Add a hexagonal obstacle to the map."""
        resolution = self.config['map_info']['resolution']
        
        center = obstacle['center']
        radius = obstacle['radius']
        angle = obstacle.get('angle', 0.0)  # Rotation angle in radians
        filled = obstacle.get('filled', True)
        
        # Convert to grid coordinates
        center_x = int(center[0] / resolution)
        center_y = int(center[1] / resolution)
        radius_cells = int(radius / resolution)
        
        # Generate hexagon vertices
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3  # 60 degrees between vertices
            x = center_x + radius_cells * math.cos(theta)
            y = center_y + radius_cells * math.sin(theta)
            vertices.append([int(x), int(y)])
        
        if filled:
            self._fill_polygon(map_data, vertices)
        else:
            # Create hexagon outline
            for i in range(6):
                start_vertex = vertices[i]
                end_vertex = vertices[(i + 1) % 6]
                self._draw_line(map_data, start_vertex, end_vertex)
        
        return map_data
    
    def _add_polygon_obstacle(self, map_data: np.ndarray, obstacle: Dict) -> np.ndarray:
        """Add a polygon obstacle to the map."""
        resolution = self.config['map_info']['resolution']
        points = obstacle['points']
        filled = obstacle.get('filled', True)
        
        # Convert points to grid coordinates
        grid_points = []
        for point in points:
            x = int(point[0] / resolution)
            y = int(point[1] / resolution)
            grid_points.append([x, y])
        
        if len(grid_points) >= 3:
            if filled:
                self._fill_polygon(map_data, grid_points)
            else:
                # Create polygon outline
                for i in range(len(grid_points)):
                    start_point = grid_points[i]
                    end_point = grid_points[(i + 1) % len(grid_points)]
                    self._draw_line(map_data, start_point, end_point)
        
        return map_data
    
    def _add_line_obstacle(self, map_data: np.ndarray, obstacle: Dict) -> np.ndarray:
        """Add a line obstacle to the map."""
        resolution = self.config['map_info']['resolution']
        
        start = obstacle['start']
        end = obstacle['end']
        width = obstacle.get('width', resolution)
        
        # Convert to grid coordinates
        start_x = int(start[0] / resolution)
        start_y = int(start[1] / resolution)
        end_x = int(end[0] / resolution)
        end_y = int(end[1] / resolution)
        width_cells = max(1, int(width / resolution))
        
        # Use Bresenham's line algorithm to draw the line
        try:
            from maps.map_utils import bresenham2D
            line_points = bresenham2D(start_x, start_y, end_x, end_y)
            
            # Add thickness to the line
            for i in range(line_points.shape[1]):
                x, y = line_points[0, i], line_points[1, i]
                # Add width around the line
                for dx in range(-width_cells//2, width_cells//2 + 1):
                    for dy in range(-width_cells//2, width_cells//2 + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            map_data[nx, ny] = 1.0
        except ImportError:
            # Fallback to simple line drawing
            self._draw_line(map_data, [start_x, start_y], [end_x, end_y], width_cells)
        
        return map_data
    
    def _add_maze_pattern(self, map_data: np.ndarray, pattern: Dict) -> np.ndarray:
        """Add a maze-like pattern to the map."""
        resolution = self.config['map_info']['resolution']
        
        cell_size = pattern.get('cell_size', 5.0)  # Size of each maze cell
        wall_thickness = pattern.get('wall_thickness', 0.5)
        gaps = pattern.get('gaps', [])  # List of gap positions
        
        cell_size_cells = int(cell_size / resolution)
        wall_thickness_cells = max(1, int(wall_thickness / resolution))
        
        # Create horizontal walls
        for i in range(cell_size_cells, self.grid_width - cell_size_cells, cell_size_cells):
            # Check if this wall should have a gap
            wall_has_gap = any(gap['type'] == 'horizontal' and gap['position'] == i for gap in gaps)
            if not wall_has_gap:
                map_data[i:i+wall_thickness_cells, cell_size_cells:self.grid_height-cell_size_cells] = 1.0
        
        # Create vertical walls
        for j in range(cell_size_cells, self.grid_height - cell_size_cells, cell_size_cells):
            # Check if this wall should have a gap
            wall_has_gap = any(gap['type'] == 'vertical' and gap['position'] == j for gap in gaps)
            if not wall_has_gap:
                map_data[cell_size_cells:self.grid_width-cell_size_cells, j:j+wall_thickness_cells] = 1.0
        
        return map_data
    
    def _add_random_obstacles(self, map_data: np.ndarray, pattern: Dict) -> np.ndarray:
        """Add random obstacles to the map."""
        num_obstacles = pattern.get('num_obstacles', 10)
        min_size = pattern.get('min_size', 1.0)
        max_size = pattern.get('max_size', 3.0)
        seed = pattern.get('seed', None)
        resolution = self.config['map_info']['resolution']
        
        if seed is not None:
            np.random.seed(seed)
        
        for _ in range(num_obstacles):
            # Random position
            x = np.random.uniform(2.0, self.config['map_info']['width'] - 2.0)
            y = np.random.uniform(2.0, self.config['map_info']['height'] - 2.0)
            
            # Random size
            size = np.random.uniform(min_size, max_size)
            
            # Convert to grid coordinates
            center_x = int(x / resolution)
            center_y = int(y / resolution)
            size_cells = int(size / resolution)
            
            # Add obstacle
            x1 = max(0, center_x - size_cells)
            x2 = min(self.grid_width, center_x + size_cells)
            y1 = max(0, center_y - size_cells)
            y2 = min(self.grid_height, center_y + size_cells)
            map_data[x1:x2, y1:y2] = 1.0
        
        return map_data
    
    def _fill_triangle(self, map_data: np.ndarray, vertices: List[List[int]]) -> None:
        """Fill a triangle using barycentric coordinates."""
        # Find bounding box
        min_x = max(0, min(v[0] for v in vertices))
        max_x = min(self.grid_width - 1, max(v[0] for v in vertices))
        min_y = max(0, min(v[1] for v in vertices))
        max_y = min(self.grid_height - 1, max(v[1] for v in vertices))
        
        # Fill triangle using barycentric coordinates
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if self._point_in_triangle(x, y, vertices):
                    map_data[x, y] = 1.0
    
    def _point_in_triangle(self, x: int, y: int, vertices: List[List[int]]) -> bool:
        """Check if a point is inside a triangle using barycentric coordinates."""
        if len(vertices) != 3:
            return False
        
        v0 = vertices[0]
        v1 = vertices[1]
        v2 = vertices[2]
        
        # Barycentric coordinates
        denominator = ((v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
        if denominator == 0:
            return False
        
        a = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denominator
        b = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denominator
        c = 1.0 - a - b
        
        return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1
    
    def _draw_triangle_outline(self, map_data: np.ndarray, vertices: List[List[int]]) -> None:
        """Draw the outline of a triangle."""
        for i in range(3):
            start_vertex = vertices[i]
            end_vertex = vertices[(i + 1) % 3]
            self._draw_line(map_data, start_vertex, end_vertex)
    
    def _fill_polygon(self, map_data: np.ndarray, vertices: List[List[int]]) -> None:
        """Fill a polygon using scan line algorithm."""
        if len(vertices) < 3:
            return
        
        # Find bounding box
        min_x = max(0, min(v[0] for v in vertices))
        max_x = min(self.grid_width - 1, max(v[0] for v in vertices))
        min_y = max(0, min(v[1] for v in vertices))
        max_y = min(self.grid_height - 1, max(v[1] for v in vertices))
        
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
                    x2 = min(self.grid_width - 1, intersections[i + 1])
                    map_data[x1:x2 + 1, y] = 1.0
    
    def _draw_line(self, map_data: np.ndarray, start: List[int], end: List[int], width: int = 1) -> None:
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
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        map_data[nx, ny] = 1.0
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    def generate_map(self) -> np.ndarray:
        """Generate the complete map based on the YAML configuration."""
        # Create empty map
        self.map_data = self._create_empty_map()
        
        # Add obstacles
        obstacles = self.config.get('obstacles', [])
        
        for obstacle in obstacles:
            obstacle_type = obstacle['type']
            
            if obstacle_type == 'rectangle':
                self.map_data = self._add_rectangular_obstacle(self.map_data, obstacle)
            elif obstacle_type == 'circle':
                self.map_data = self._add_circular_obstacle(self.map_data, obstacle)
            elif obstacle_type == 'triangle':
                self.map_data = self._add_triangular_obstacle(self.map_data, obstacle)
            elif obstacle_type == 'hexagon':
                self.map_data = self._add_hexagonal_obstacle(self.map_data, obstacle)
            elif obstacle_type == 'polygon':
                self.map_data = self._add_polygon_obstacle(self.map_data, obstacle)
            elif obstacle_type == 'line':
                self.map_data = self._add_line_obstacle(self.map_data, obstacle)
            elif obstacle_type == 'maze':
                self.map_data = self._add_maze_pattern(self.map_data, obstacle)
            elif obstacle_type == 'random':
                self.map_data = self._add_random_obstacles(self.map_data, obstacle)
            else:
                print(f"Warning: Unknown obstacle type '{obstacle_type}'")
        
        return self.map_data
    
    def save_map_files(self, output_dir: str = None) -> None:
        """Save all map files (YAML, CFG) to the specified directory."""
        if self.map_data is None:
            self.generate_map()
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(self.yaml_path)
            # If the YAML file is in the current directory, use current directory
            if not output_dir:
                output_dir = '.'
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get map name from YAML file
        map_name = Path(self.yaml_path).stem
        
        # Save CFG file
        cfg_path = os.path.join(output_dir, f"{map_name}.cfg")
        np.savetxt(cfg_path, self.map_data, fmt='%d')
        print(f"Map data saved to: {cfg_path}")
        
        # Create and save YAML configuration file
        yaml_config = self._create_yaml_config(map_name)
        yaml_path = os.path.join(output_dir, f"{map_name}.yaml")
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
        print(f"YAML configuration saved to: {yaml_path}")
        
        # Save visualization
        self.save_visualization(output_dir, map_name)
    
    def _create_yaml_config(self, map_name: str) -> Dict:
        """Create the YAML configuration for the generated map."""
        map_info = self.config['map_info']
        
        yaml_config = {
            'datatype': 't',
            'mapdim': [self.grid_width, self.grid_height],
            'mapmax': [map_info['width'], map_info['height']],
            'mapmin': [0.0, 0.0],
            'mappath': f"{map_name}.cfg",
            'mapres': [map_info['resolution'], map_info['resolution']],
            'origin': [map_info['width']/2, map_info['height']/2],
            'origincells': [self.grid_width//2, self.grid_height//2],
            'storage': 'colmajor'
        }
        
        return yaml_config
    
    def save_visualization(self, output_dir: str, map_name: str) -> None:
        """Save a visualization of the generated map."""
        if self.map_data is None:
            self.generate_map()
        
        plt.figure(figsize=(12, 10))
        plt.imshow(self.map_data, cmap='gray_r', origin='lower')
        plt.title(f'Generated Map: {map_name}')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.colorbar(label='Occupancy (0=Free, 1=Occupied)')
        
        # Add grid lines
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_path = os.path.join(output_dir, f"{map_name}_visualization.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {plot_path}")
    
    def preview_map(self) -> None:
        """Preview the generated map."""
        if self.map_data is None:
            self.generate_map()
        
        plt.figure(figsize=(12, 10))
        plt.imshow(self.map_data, cmap='gray_r', origin='lower')
        plt.title(f'Map Preview: {Path(self.yaml_path).stem}')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.colorbar(label='Occupancy (0=Free, 1=Occupied)')
        plt.grid(True, alpha=0.3)
        plt.show()

def create_sample_yaml_config(map_name: str = "sample_map") -> str:
    """Create a sample YAML configuration file."""
    sample_config = f"""# Sample Map Configuration for {map_name}
# This file defines all parameters for map generation

map_info:
  name: "{map_name}"
  width: 72.4          # Map width in meters
  height: 72.4         # Map height in meters
  resolution: 0.4      # Grid resolution in meters
  boundary:
    enabled: true      # Whether to add boundary walls
    thickness: 0.4     # Boundary wall thickness in meters
    style: "rectangle" # Boundary style: "rectangle", "circle", "polygon"
    rectangle:
      margin: 0.0      # Margin from map edges in meters

obstacles:
  # Rectangular obstacles
  - type: "rectangle"
    center: [20.0, 20.0]     # Center coordinates [x, y] in meters
    width: 10.0              # Width in meters
    height: 5.0              # Height in meters
    angle: 0.0               # Rotation angle in radians (0 = no rotation)
    filled: true             # Whether the rectangle is filled
  
  # Circular obstacles
  - type: "circle"
    center: [10.0, 10.0]     # Center coordinates [x, y] in meters
    radius: 3.0              # Radius in meters
    filled: true             # Whether the circle is filled
  
  # Triangular obstacles
  - type: "triangle"
    vertices: [[15.0, 50.0], [25.0, 50.0], [20.0, 60.0]]  # Three vertices [x, y]
    filled: true
  
  # Hexagonal obstacles
  - type: "hexagon"
    center: [30.0, 30.0]     # Center coordinates [x, y] in meters
    radius: 4.0              # Radius (distance from center to vertices)
    angle: 0.0               # Rotation angle in radians
    filled: true
  
  # Line obstacles
  - type: "line"
    start: [0.0, 30.0]       # Start point [x, y] in meters
    end: [50.0, 30.0]        # End point [x, y] in meters
    width: 0.5               # Line width in meters
  
  # Random obstacles
  - type: "random"
    num_obstacles: 5         # Number of random obstacles
    min_size: 1.0           # Minimum obstacle size in meters
    max_size: 3.0           # Maximum obstacle size in meters
    seed: 42                # Random seed for reproducibility
"""
    
    return sample_config

def main():
    """Main function to run the map generator."""
    parser = argparse.ArgumentParser(description='Advanced Map Generator for AJLATT Environment')
    parser.add_argument('yaml_file', help='Path to the YAML configuration file')
    parser.add_argument('--output-dir', '-o', help='Output directory for generated files')
    parser.add_argument('--preview', '-p', action='store_true', help='Preview the generated map')
    parser.add_argument('--create-sample', '-s', help='Create a sample YAML configuration file')
    
    args = parser.parse_args()
    
    if args.create_sample:
        # Create a sample YAML file
        sample_config = create_sample_yaml_config(args.create_sample)
        sample_path = f"{args.create_sample}.yaml"
        with open(sample_path, 'w') as f:
            f.write(sample_config)
        print(f"Sample YAML configuration created: {sample_path}")
        return
    
    try:
        # Initialize the map generator
        generator = AdvancedMapGenerator(args.yaml_file)
        
        # Generate the map
        print(f"Generating map from: {args.yaml_file}")
        generator.generate_map()
        
        # Preview if requested
        if args.preview:
            generator.preview_map()
        
        # Save all files
        generator.save_map_files(args.output_dir)
        
        print("Map generation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()