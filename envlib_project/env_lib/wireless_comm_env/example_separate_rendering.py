#!/usr/bin/env python3
"""
Example showing separate control of real-time rendering and GIF saving.
"""

import os
import time
import numpy as np
from wireless_comm_env import WirelessCommEnv
import matplotlib.pyplot as plt


def run_with_separate_rendering_control(
    grid_x=4,
    grid_y=4,
    steps=20,
    enable_realtime=True,
    enable_gif=True,
    gif_path=None,
    frame_delay=1.5
):
    """
    Run the environment with separate control over real-time rendering and GIF saving.
    
    Args:
        grid_x, grid_y: Grid size for agents
        steps: Number of steps to run
        enable_realtime: Whether to show real-time rendering
        enable_gif: Whether to save frames as GIF
        gif_path: Path to save the GIF
        frame_delay: Delay between frames for real-time viewing
    """
    print(f"Starting environment with separate rendering control...")
    print(f"Grid: {grid_x}x{grid_y}, Steps: {steps}")
    print(f"Real-time rendering: {enable_realtime}")
    print(f"GIF saving: {enable_gif}")
    
    # Initialize environment
    env = WirelessCommEnv(grid_x=grid_x, grid_y=grid_y, render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up real-time display if enabled
    if enable_realtime:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        ax.set_title('Wireless Communication Environment', fontsize=14, fontweight='bold')
        im = ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        plt.tight_layout()
    
    # Start frame collection if GIF saving is enabled
    if enable_gif:
        env.start_frame_collection()
    
    # Run the environment
    for step in range(steps):
        print(f"Step {step + 1}/{steps}")
        
        # Take action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render (this will collect frames if enabled)
        frame = env.render()
        
        # Update real-time display
        if enable_realtime:
            im.set_data(frame)
            ax.set_title(f'Wireless Communication Environment - Step {step + 1}', 
                        fontsize=14, fontweight='bold')
            plt.draw()
            plt.pause(frame_delay)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # Clean up real-time display
    if enable_realtime:
        plt.ioff()  # Turn off interactive mode
        plt.close(fig)
    
    # Stop frame collection and save GIF
    if enable_gif:
        env.stop_frame_collection()
        if gif_path is None:
            gif_path = os.path.join(os.getcwd(), 'wireless_comm_separate.gif')
        env.save_gif(gif_path=gif_path, fps=3)
    
    env.close()
    print("Environment run completed!")


def run_realtime_only():
    """Run with only real-time rendering (no GIF saving)."""
    print("Running with real-time rendering only...")
    run_with_separate_rendering_control(
        grid_x=4,
        grid_y=4,
        steps=150,
        enable_realtime=True,
        enable_gif=False,
        frame_delay=0.8
    )


def run_gif_only():
    """Run with only GIF saving (no real-time display)."""
    print("Running with GIF saving only...")
    run_with_separate_rendering_control(
        grid_x=4,
        grid_y=4,
        steps=20,
        enable_realtime=False,
        enable_gif=True,
        gif_path="wireless_comm_gif_only.gif",
        frame_delay=0.1  # Fast processing since no display
    )


def run_both_modes():
    """Run with both real-time rendering and GIF saving."""
    print("Running with both real-time rendering and GIF saving...")
    run_with_separate_rendering_control(
        grid_x=4,
        grid_y=4,
        steps=15,
        enable_realtime=True,
        enable_gif=True,
        gif_path="wireless_comm_both_modes.gif",
        frame_delay=0.6
    )


if __name__ == "__main__":
    print("Wireless Communication Environment - Separate Rendering Control")
    print("=" * 60)
    print("Choose rendering mode:")
    print("1. Real-time rendering only")
    print("2. GIF saving only")
    print("3. Both real-time and GIF")
    print("4. Custom configuration")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        run_realtime_only()
    elif choice == "2":
        run_gif_only()
    elif choice == "3":
        run_both_modes()
    elif choice == "4":
        # Custom configuration
        grid_x = int(input("Enter grid_x (default 4): ") or "4")
        grid_y = int(input("Enter grid_y (default 4): ") or "4")
        steps = int(input("Enter number of steps (default 15): ") or "15")
        realtime = input("Enable real-time rendering? (y/n, default y): ").lower() != 'n'
        gif = input("Enable GIF saving? (y/n, default y): ").lower() != 'n'
        
        run_with_separate_rendering_control(
            grid_x=grid_x,
            grid_y=grid_y,
            steps=steps,
            enable_realtime=realtime,
            enable_gif=gif,
            frame_delay=0.6
        )
    else:
        print("Invalid choice. Running default configuration...")
        run_both_modes() 