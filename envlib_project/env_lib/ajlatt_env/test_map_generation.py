#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : test_map_generation.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: RL_LLM
@Date   : 10/05/2024
@Time   : 18:56:07
@Info   : Test script for the Advanced Map Generation System
"""

import os
import sys
import yaml
import numpy as np

# Add the env directory to the path
sys.path.append(os.path.dirname(__file__))

from map_generate import AdvancedMapGenerator, create_sample_yaml_config

def test_basic_map_generation():
    """Test basic map generation functionality."""
    print("=== Testing Basic Map Generation ===")
    
    # Create a simple YAML configuration
    simple_config = {
        'map_info': {
            'name': 'test_map',
            'width': 50.0,
            'height': 50.0,
            'resolution': 0.5,
            'add_boundary': True
        },
        'obstacles': [
            {
                'type': 'rectangle',
                'center': [25.0, 25.0],
                'width': 10.0,
                'height': 5.0,
                'angle': 0.0
            },
            {
                'type': 'circle',
                'center': [10.0, 10.0],
                'radius': 3.0
            }
        ]
    }
    
    # Save the configuration
    with open('test_config.yaml', 'w') as f:
        yaml.dump(simple_config, f, default_flow_style=False)
    
    try:
        # Generate the map
        generator = AdvancedMapGenerator('test_config.yaml')
        map_data = generator.generate_map()
        
        print(f"Map generated with shape: {map_data.shape}")
        print(f"Number of occupied cells: {np.sum(map_data)}")
        
        # Save the map
        generator.save_map_files()
        
        print("Basic map generation test completed!")
        
    finally:
        # Clean up files
        cleanup_files = [
            'test_config.yaml',
            'test_map.yaml',
            'test_map.cfg',
            'test_map_visualization.png'
        ]
        for file in cleanup_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Cleaned up: {file}")

def test_advanced_obstacles():
    """Test advanced obstacle types."""
    print("\n=== Testing Advanced Obstacles ===")
    
    # Create a configuration with various obstacle types
    advanced_config = {
        'map_info': {
            'name': 'advanced_test',
            'width': 60.0,
            'height': 60.0,
            'resolution': 0.4,
            'add_boundary': True
        },
        'obstacles': [
            # Rectangular obstacles with rotation
            {
                'type': 'rectangle',
                'center': [20.0, 20.0],
                'width': 8.0,
                'height': 4.0,
                'angle': 0.785  # 45 degrees
            },
            # Circular obstacles
            {
                'type': 'circle',
                'center': [40.0, 40.0],
                'radius': 5.0
            },
            # Line obstacles
            {
                'type': 'line',
                'start': [0.0, 30.0],
                'end': [40.0, 30.0],
                'width': 0.8
            },
            # Maze pattern
            {
                'type': 'maze',
                'cell_size': 6.0,
                'wall_thickness': 0.5,
                'gaps': [
                    {'type': 'horizontal', 'position': 15},
                    {'type': 'vertical', 'position': 20}
                ]
            },
            # Random obstacles
            {
                'type': 'random',
                'num_obstacles': 5,
                'min_size': 1.0,
                'max_size': 2.5
            }
        ]
    }
    
    # Save the configuration
    with open('advanced_test_config.yaml', 'w') as f:
        yaml.dump(advanced_config, f, default_flow_style=False)
    
    try:
        # Generate the map
        generator = AdvancedMapGenerator('advanced_test_config.yaml')
        map_data = generator.generate_map()
        
        print(f"Advanced map generated with shape: {map_data.shape}")
        print(f"Number of occupied cells: {np.sum(map_data)}")
        
        # Save the map
        generator.save_map_files()
        
        print("Advanced obstacles test completed!")
        
    finally:
        # Clean up files
        cleanup_files = [
            'advanced_test_config.yaml',
            'advanced_test.yaml',
            'advanced_test.cfg',
            'advanced_test_visualization.png'
        ]
        for file in cleanup_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Cleaned up: {file}")

def test_sample_config():
    """Test using the sample configuration file."""
    print("\n=== Testing Sample Configuration ===")
    
    # Create sample configuration
    sample_config = create_sample_yaml_config("sample_test")
    with open('sample_test.yaml', 'w') as f:
        f.write(sample_config)
    
    try:
        # Generate the map
        generator = AdvancedMapGenerator('sample_test.yaml')
        map_data = generator.generate_map()
        
        print(f"Sample map generated with shape: {map_data.shape}")
        print(f"Number of occupied cells: {np.sum(map_data)}")
        
        # Save the map
        generator.save_map_files()
        
        print("Sample configuration test completed!")
        
    finally:
        # Clean up files
        cleanup_files = [
            'sample_test.yaml',
            'sample_test_visualization.png'
        ]
        for file in cleanup_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Cleaned up: {file}")

def test_environment_integration():
    """Test integration with the AJLATT environment."""
    print("\n=== Testing Environment Integration ===")
    
    try:
        import env_lib
        
        # Generate a simple map first
        simple_config = {
            'map_info': {
                'name': 'env_test',
                'width': 72.4,
                'height': 72.4,
                'resolution': 0.4,
                'add_boundary': True
            },
            'obstacles': [
                {
                    'type': 'rectangle',
                    'center': [30.0, 30.0],
                    'width': 15.0,
                    'height': 8.0,
                    'angle': 0.0
                },
                {
                    'type': 'circle',
                    'center': [50.0, 50.0],
                    'radius': 6.0
                }
            ]
        }
        
        with open('env_test_config.yaml', 'w') as f:
            yaml.dump(simple_config, f, default_flow_style=False)
        
        # Generate the map
        generator = AdvancedMapGenerator('env_test_config.yaml')
        generator.save_map_files()
        
        # Test with the environment
        env = env_lib.ajlatt_env(map_name='env_test')
        obs = env.reset()
        
        print("Environment integration test successful!")
        print(f"Environment observation shape: {obs.shape}")
        
        # Clean up
        os.remove('env_test_config.yaml')
        os.remove('env_test.yaml')
        os.remove('env_test.cfg')
        os.remove('env_test_visualization.png')
        
    except ImportError:
        print("env_lib not available, skipping environment integration test")
    except Exception as e:
        print(f"Environment integration test failed: {e}")

def main():
    """Run all tests."""
    print("Starting Map Generation Tests...\n")
    
    try:
        test_basic_map_generation()
        test_advanced_obstacles()
        test_sample_config()
        test_environment_integration()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 