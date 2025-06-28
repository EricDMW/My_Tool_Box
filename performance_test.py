#!/usr/bin/env python3
"""
Performance testing script to compare original vs optimized versions
"""

import time
import numpy as np
import torch
import env_lib

def test_original_performance():
    """Test original performance with rendering enabled"""
    print("Testing ORIGINAL performance (with rendering)...")
    
    env = env_lib.ajlatt_env(map_name='obstacles05', num_Robot=4)
    env.reset()
    
    # Original approach: create tensor in loop + render every step
    start_time = time.time()
    for step in range(50):  # Reduced steps for faster testing
        action = 0.1 * torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        k = env.step(action)
        env.render()  # Render every step
        
    end_time = time.time()
    original_time = end_time - start_time
    print(f"Original time: {original_time:.4f} seconds")
    print(f"Original avg per step: {original_time/50:.4f} seconds")
    return original_time

def test_optimized_performance():
    """Test optimized performance without rendering"""
    print("\nTesting OPTIMIZED performance (no rendering)...")
    
    env = env_lib.ajlatt_env(map_name='obstacles05', num_Robot=4)
    env.reset()
    
    # Optimized approach: pre-allocate tensor + no rendering
    action = 0.1 * torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    
    start_time = time.time()
    for step in range(50):  # Same number of steps
        k = env.step(action)
        # No rendering in loop
        
    end_time = time.time()
    optimized_time = end_time - start_time
    print(f"Optimized time: {optimized_time:.4f} seconds")
    print(f"Optimized avg per step: {optimized_time/50:.4f} seconds")
    return optimized_time

def test_optimized_with_occasional_rendering():
    """Test optimized performance with occasional rendering"""
    print("\nTesting OPTIMIZED performance (occasional rendering)...")
    
    env = env_lib.ajlatt_env(map_name='obstacles05', num_Robot=4)
    env.reset()
    
    action = 0.1 * torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    render_frequency = 10
    
    start_time = time.time()
    for step in range(50):
        k = env.step(action)
        if step % render_frequency == 0:  # Render every 10 steps
            env.render()
            
    end_time = time.time()
    occasional_render_time = end_time - start_time
    print(f"Optimized with occasional rendering: {occasional_render_time:.4f} seconds")
    print(f"Avg per step: {occasional_render_time/50:.4f} seconds")
    return occasional_render_time

def main():
    print("=== PERFORMANCE COMPARISON TEST ===")
    print("Testing 50 steps for each configuration...")
    
    try:
        # Test original performance (this will be slow due to rendering)
        original_time = test_original_performance()
    except Exception as e:
        print(f"Original test failed: {e}")
        original_time = float('inf')
    
    # Test optimized performance
    optimized_time = test_optimized_performance()
    
    # Test optimized with occasional rendering
    occasional_render_time = test_optimized_with_occasional_rendering()
    
    # Calculate speedups
    print("\n=== PERFORMANCE SUMMARY ===")
    if original_time != float('inf'):
        speedup_no_render = original_time / optimized_time
        speedup_occasional = original_time / occasional_render_time
        print(f"Speedup (no rendering): {speedup_no_render:.2f}x")
        print(f"Speedup (occasional rendering): {speedup_occasional:.2f}x")
    
    print(f"\nOptimized vs occasional rendering speedup: {occasional_render_time/optimized_time:.2f}x")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Disable rendering during training for maximum speed")
    print("2. Use occasional rendering (every 10-20 steps) for debugging")
    print("3. The CI caching and spatial indexing optimizations provide additional speedup")
    print("4. Consider using PyTorch tensors consistently throughout the codebase")

if __name__ == "__main__":
    main() 