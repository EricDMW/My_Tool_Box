#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : test_build_map.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: RL_LLM
@Date   : 10/05/2024
@Time   : 18:56:07
@Info   : Test script for build_map.py main function
"""

from build_map import main

if __name__ == "__main__":
    # Example 1: Basic usage with YAML file path
    print("=== Example 1: Basic map generation ===")
    main("env/maps/sample_map_config.yaml")
    
    # Example 2: With analysis enabled
    print("\n=== Example 2: With detailed analysis ===")
    main("env/maps/sample_map_config.yaml", analysis=True)
    
    # Example 3: Preview mode (if implemented)
    print("\n=== Example 3: Preview mode ===")
    main("env/maps/sample_map_config.yaml", preview=True) 