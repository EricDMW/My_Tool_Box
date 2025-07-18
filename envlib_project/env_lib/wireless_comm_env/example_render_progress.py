import os
import time
import numpy as np
from wireless_comm_env import WirelessCommEnv
import matplotlib.pyplot as plt
from matplotlib import animation


def run_and_render_animation_realtime(
    grid_x=4,
    grid_y=4,
    steps=30,
    save_gif=False,
    gif_path=None,
    show=True,
    frame_delay=0.5
):
    """
    Run the wireless communication environment and render in real-time during progress.
    Args:
        grid_x, grid_y: Grid size for agents
        steps: Number of steps to animate
        save_gif: Whether to save the animation as a GIF
        gif_path: Path to save the GIF (default: current working directory)
        show: Whether to display the animation
        frame_delay: Delay between frames in seconds for smooth animation
    """
    print(f"Starting real-time animation with {grid_x}x{grid_y} grid, {steps} steps...")
    
    # Set up interactive mode for real-time display
    plt.ion()  # Turn on interactive mode
    
    env = WirelessCommEnv(grid_x=grid_x, grid_y=grid_y, render_mode="rgb_array")
    obs, info = env.reset()
    frames = []
    
    # Create figure for real-time display
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    ax.set_title('Wireless Communication Environment - Real-time Progress', fontsize=14, fontweight='bold')
    
    # Initialize display
    im = ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
    plt.tight_layout()
    
    for step in range(steps):
        print(f"Processing step {step + 1}/{steps}...")
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env._render_rgb_array()
        frames.append(frame)
        
        # Update display in real-time
        im.set_data(frame)
        ax.set_title(f'Wireless Communication Environment - Step {step + 1}', fontsize=14, fontweight='bold')
        plt.draw()
        plt.pause(frame_delay)  # This allows real-time viewing
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print(f"Generated {len(frames)} frames")

    # Save GIF if requested
    if save_gif:
        if gif_path is None:
            gif_path = os.path.join(os.getcwd(), 'wireless_comm_realtime.gif')
        
        print(f"Saving animation to {gif_path}...")
        # Create animation from collected frames
        anim = animation.FuncAnimation(fig, lambda i: [im.set_data(frames[i])], 
                                     frames=len(frames), interval=int(frame_delay*1000), 
                                     blit=False, repeat=True)
        fps = max(2, int(1.0 / frame_delay))
        anim.save(gif_path, writer='pillow', fps=fps, dpi=100)
        print(f"Animation saved successfully!")

    if show:
        plt.ioff()  # Turn off interactive mode
        plt.show()
    else:
        plt.close(fig)

    return frames


def run_simple_realtime_demo():
    """Run a simple real-time demonstration."""
    print("Running simple real-time wireless communication demo...")
    
    run_and_render_animation_realtime(
        grid_x=3,
        grid_y=3,
        steps=15,
        save_gif=True,
        gif_path="wireless_comm_simple_realtime.gif",
        show=True,
        frame_delay=0.8  # Slower for easier viewing
    )


if __name__ == "__main__":
    # Example usage with real-time rendering
    print("Choose rendering mode:")
    print("1. Real-time rendering (see progress live)")
    print("2. Simple demo (3x3 grid, 15 steps)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_and_render_animation_realtime(
            grid_x=4,
            grid_y=4,
            steps=20,
            save_gif=True,
            gif_path="wireless_comm_realtime.gif",
            show=True,
            frame_delay=0.6  # Real-time with 0.6s delay
        )
    else:
        run_simple_realtime_demo() 