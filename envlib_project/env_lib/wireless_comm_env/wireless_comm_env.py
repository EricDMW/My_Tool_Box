"""
Wireless Communication Environment - A standard gym environment for multi-agent wireless communication.

This environment simulates a wireless communication network where agents must coordinate
to transmit packets through access points while avoiding interference.

Parameters:
- grid_x: Number of agents in x-direction (default: 6)
- grid_y: Number of agents in y-direction (default: 6)
- ddl: Deadline horizon for packet transmission (default: 2)
- packet_arrival_probability: Probability of new packet arrival (default: 0.8)
- success_transmission_probability: Probability of successful transmission (default: 0.8)
- n_obs_neighbors: Number of neighbors to observe (default: 1)
- max_iter: Maximum number of steps per episode (default: 50)
- render_mode: Rendering mode ('human', 'rgb_array', None)
"""

import copy
import os
import numpy as np
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.utils import EzPickle, seeding
import torch


class WirelessCommEnv(Env, EzPickle):
    """
    Wireless Communication Environment - Standard gym environment for multi-agent wireless communication.
    
    This environment simulates a wireless communication network where agents must coordinate
    to transmit packets through access points while avoiding interference.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "WirelessComm-v0",
        "render_fps": 10,
    }

    def __init__(
        self,
        grid_x=6,
        grid_y=6,
        ddl=2,
        packet_arrival_probability=0.8,
        success_transmission_probability=0.8,
        n_obs_neighbors=1,
        max_iter=50,
        render_mode=None,
        save_gif=False,
        gif_path=None,
        debug_info=False,
        device='cpu',
    ):
        """
        Initialize the Wireless Communication environment.
        
        Args:
            grid_x: Number of agents in x-direction
            grid_y: Number of agents in y-direction
            ddl: Deadline horizon for packet transmission
            packet_arrival_probability: Probability of new packet arrival
            success_transmission_probability: Probability of successful transmission
            n_obs_neighbors: Number of neighbors to observe
            max_iter: Maximum number of steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', None)
            save_gif: If True, save a GIF of the animation when env.close() is called.
            gif_path: Path to save the GIF (default: wireless_comm_run.gif)
            debug_info: If True, info dict in step() contains neighbors and access_points; otherwise only local_rewards.
            device: Device to use for torch tensors (e.g., 'cpu', 'cuda')
        """
        EzPickle.__init__(
            self,
            grid_x,
            grid_y,
            ddl,
            packet_arrival_probability,
            success_transmission_probability,
            n_obs_neighbors,
            max_iter,
            render_mode,
        )
        
        # Environment parameters
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.ddl = ddl
        self.p = packet_arrival_probability
        self.q = success_transmission_probability
        self.n_obs_nghbr = n_obs_neighbors
        self.max_iter = max_iter
        self.render_mode = render_mode
        self.save_gif = save_gif
        self.gif_path = gif_path or os.path.join(os.getcwd(), 'wireless_comm_run.gif')
        self._mpl_fig = None
        self._mpl_ax = None
        self._mpl_im = None
        
        # Agent setup
        self.n_agents = grid_x * grid_y
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_agents))))
        
        # Action and observation spaces - matching original PettingZoo structure
        self.action_space = MultiDiscrete([5] * self.n_agents)
        
        # Observation space: each agent gets its local state (matching original)
        obs_dim = self.ddl * ((self.n_obs_nghbr * 2 + 1) ** 2)
        self.observation_space = Box(
            low=0, 
            high=2, 
            shape=(self.n_agents, obs_dim), 
            dtype=np.float32
        )
        
        # Game state - matching original structure
        self.state = None
        self.actions = None
        self.num_moves = 0
        self.frames = []
        self.collecting_frames = False
        
        self.closed = False
        self.seed()
        self.debug_info = debug_info
        self.device = torch.device(device)
        
        # Compute neighbors for each agent
        self.neighbors = []
        for agent_id in range(self.n_agents):
            agent_x = agent_id % self.grid_x
            agent_y = agent_id // self.grid_x
            nghbrs = []
            # Only consider up, down, left, right
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = agent_x + dx, agent_y + dy
                if 0 <= nx < self.grid_x and 0 <= ny < self.grid_y:
                    neighbor_id = ny * self.grid_x + nx
                    nghbrs.append(neighbor_id)
            self.neighbors.append(nghbrs)

        # Access points for each agent (as list of possible AP indices)
        self.access_points = []
        for agent_id in range(self.n_agents):
            agent_x = agent_id % self.grid_x
            agent_y = agent_id // self.grid_x
            ap_indices = []
            # Check all four possible APs
            for action in [1, 2, 3, 4]:  # LL, UL, LR, UR
                ap_x, ap_y = self.access_point_mapping(agent_x, agent_y, action)
                if ap_x is not None and ap_y is not None:
                    ap_id = ap_y * (self.grid_x - 1) + ap_x + 1  # 1-based index
                    ap_indices.append(ap_id)
            self.access_points.append(ap_indices)

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def access_point_mapping(self, x, y, agent_action):
        """Map agent action to access point coordinates (UL, LL, UR, LR) for (x, y) with y increasing upwards."""
        if agent_action == 0:
            return None, None
        # LL: lower and left
        if agent_action == 1:
            ap_x, ap_y = x - 1, y
        # UL: upper and left
        elif agent_action == 2:
            ap_x, ap_y = x - 1, y - 1
        # LR: lower and right
        elif agent_action == 3:
            ap_x, ap_y = x, y
        # UR: upper and right
        elif agent_action == 4:
            ap_x, ap_y = x, y - 1
        else:
            raise ValueError(f'agent_action = {agent_action} is not defined!')
        # Only valid if in bounds
        if 0 <= ap_x < self.grid_x - 1 and 0 <= ap_y < self.grid_y - 1:
            return ap_x, ap_y
        else:
            return None, None

    def check_transmission_fail(self, agent_access_point_x, agent_access_point_y, access_point_profile):
        """Check if transmission fails due to interference (matching original logic)."""
        if agent_access_point_x is None or agent_access_point_y is None:
            return True
        return access_point_profile[agent_access_point_x, agent_access_point_y] != 1

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state (matching original logic)."""
        if seed is not None:
            self.seed(seed)
            
        # Initialize state - matching original structure
        self.state = torch.full(
            (self.ddl, self.grid_x + 2 * self.n_obs_nghbr, self.grid_y + 2 * self.n_obs_nghbr),
            fill_value=2, dtype=torch.float32, device=self.device
        )
        self.state[:, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = 0
        
        # Initialize actions
        self.actions = torch.zeros(
            (self.grid_x + 2 * self.n_obs_nghbr, self.grid_y + 2 * self.n_obs_nghbr),
            dtype=torch.int64, device=self.device
        )
        
        # Initialize packet arrivals - matching original probability
        self.state[:, self.n_obs_nghbr:self.grid_x + self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y + self.n_obs_nghbr] = \
            torch.randint(0, 2, (self.ddl, self.grid_x, self.grid_y), device=self.device)
        
        self.num_moves = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Get observations for all agents (row-major order)."""
        assert self.state is not None
        obs = torch.zeros((self.n_agents, self.ddl * ((self.n_obs_nghbr * 2 + 1) ** 2)), dtype=torch.float32, device=self.device)
        
        for i in range(self.n_agents):
            agent_x = i % self.grid_x
            agent_y = i // self.grid_x
            obs[i] = self.state[:, 
                               agent_x: agent_x + self.n_obs_nghbr * 2 + 1,
                               agent_y: agent_y + self.n_obs_nghbr * 2 + 1].reshape(-1)
        
        return obs.cpu().numpy()

    def step(self, action):
        """Efficient and consistent step logic."""
        assert self.state is not None
        action_tensor = torch.as_tensor(action, dtype=torch.int64, device=self.device).flatten()
        self.actions = torch.zeros(
            (self.grid_x + 2 * self.n_obs_nghbr, self.grid_y + 2 * self.n_obs_nghbr),
            dtype=torch.int64, device=self.device
        )
        agent_xs = torch.arange(self.n_agents, device=self.device) % self.grid_x
        agent_ys = torch.arange(self.n_agents, device=self.device) // self.grid_x
        self.actions[agent_xs + self.n_obs_nghbr, agent_ys + self.n_obs_nghbr] = action_tensor

        # Build access point profile and agent-to-ap mapping
        access_point_profile = torch.zeros((self.grid_x - 1, self.grid_y - 1), dtype=torch.int64, device=self.device)
        agent_ap = [None] * self.n_agents
        for agent_id in range(self.n_agents):
            agent_x = agent_id % self.grid_x
            agent_y = agent_id // self.grid_x
            agent_action = action_tensor[agent_id]
            ap_x, ap_y = self.access_point_mapping(agent_x, agent_y, agent_action)
            if agent_action > 0 and ap_x is not None and ap_y is not None:
                access_point_profile[ap_x, ap_y] += 1
                agent_ap[agent_id] = (ap_x, ap_y)

        rewards = torch.zeros(self.n_agents, dtype=torch.float32, device=self.device)
        for agent_id in range(self.n_agents):
            agent_x = agent_id % self.grid_x
            agent_y = agent_id // self.grid_x
            agent_action = action_tensor[agent_id]
            ap = agent_ap[agent_id]
            if agent_action > 0 and ap is not None and access_point_profile[ap[0], ap[1]] == 1:
                # Only one agent to this AP, check for packet and process
                agent_state = self.state[:, agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr]
                if agent_state.max() > 0:
                    idx = torch.argmax((agent_state > 0).to(torch.float32))
                    if self.np_random.random() <= self.q:
                        self.state[idx, agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr] = 0
                        rewards[agent_id] = 1


        # Update state: left shift and append new packets
        self.state[:-1, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = \
            self.state[1:, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr]
        self.state[-1, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = \
            (torch.rand((self.grid_x, self.grid_y), device=self.device) <= self.p).float()
        self.num_moves += 1
        self._last_rewards = rewards.cpu().numpy()
        terminated = False
        truncated = self.num_moves >= self.max_iter
        total_reward = rewards.sum().item()
        # Compute local_obs for each agent: agent's own state for all deadlines
        local_obs = []
        for agent_id in range(self.n_agents):
            agent_x = agent_id % self.grid_x
            agent_y = agent_id // self.grid_x
            # Extract the agent's own state for all deadlines
            agent_state = self.state[:, agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr]
            local_obs.append(agent_state.cpu().numpy())
        if self.debug_info:
            info = {
                "local_rewards": self._last_rewards.copy(),
                "neighbors": self.neighbors,
                "access_points": self.access_points,
                "local_obs": local_obs.copy()
            }
        else:
            info = {"local_rewards": self._last_rewards.copy(), "local_obs": local_obs}
        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self, show=True):
        """
        Render the environment. If show=True, display the current frame in a persistent matplotlib window for animation.
        If save_gif is True, collect frames for GIF saving on close().
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if show:
            if self._mpl_fig is None or self._mpl_ax is None or self._mpl_im is None:
                plt.ion()
                self._mpl_fig, self._mpl_ax = plt.subplots(figsize=(10, 10))
                self._mpl_ax.axis('off')
                # Initialize with a dummy image
                dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
                self._mpl_im = self._mpl_ax.imshow(dummy_img)
                self._mpl_fig.tight_layout()
            # Draw on the persistent axes
            rgb_array = self._render_rgb_array(ax=self._mpl_ax)
            self._mpl_im.set_data(rgb_array)
            self._mpl_ax.set_title(f'Wireless Communication Environment - Step {self.num_moves}', fontsize=14, fontweight='bold')
            self._mpl_fig.canvas.draw_idle()
            plt.pause(0.001)
        else:
            rgb_array = self._render_rgb_array()
        if self.save_gif:
            self.frames.append(rgb_array.copy())
        return rgb_array

    def _render_rgb_array(self, ax=None):
        """Create a detailed RGB array visualization of the environment."""
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyArrowPatch
        import numpy as np
        import io
        from PIL import Image

        own_fig = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            own_fig = True
        else:
            fig = None
            ax.clear()
        ax.set_xlim(-0.5, self.grid_x)
        ax.set_ylim(-0.5, self.grid_y - .4)  # Adds 1 unit of space at the bottom
        ax.set_aspect('equal')
        ax.axis('off')
        if own_fig:
            fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Colors
        agent_color = '#87CEEB'  # Light blue
        agent_edge = '#4682B4'   # Darker blue
        transmission_color = '#FF69B4'  # Pink
        general_transmission_color = '#000000'  # Black
        access_point_color = '#90EE90'  # Light green
        illegal_action_color = 'red'

        # Draw access points first (background) and number them row-major, top-to-bottom, left-to-right
        for ap_x in range(self.grid_x - 1):
            for ap_y in range(self.grid_y - 1):
                display_y = self.grid_y - 2 - ap_y
                ap_id = ap_y * (self.grid_x - 1) + ap_x + 1
                circle = plt.Circle((ap_x + 0.5, display_y + 0.5), 0.08, color=access_point_color, alpha=0.6)
                ax.add_patch(circle)
                ax.text(ap_x + 0.5, display_y + 0.5, f'{ap_id}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='green', bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))

        # Draw agents with better spacing and row-major numbering, top-to-bottom
        for agent_id in range(self.n_agents):
            agent_x = agent_id % self.grid_x
            agent_y = agent_id // self.grid_x
            display_y = self.grid_y - 1 - agent_y
            # Main cube (slightly smaller to avoid overlap)
            cube = patches.Rectangle((agent_x - 0.25, display_y - 0.25), 0.5, 0.5,
                                   linewidth=2, edgecolor=agent_edge, facecolor=agent_color, alpha=0.9)
            ax.add_patch(cube)
            # Agent ID (1-based, row-major, top-to-bottom)
            display_id = agent_id + 1
            ax.text(agent_x, display_y, f'{display_id}', ha='center', va='center',
                   fontsize=11, fontweight='bold', color='black')
            # Action (positioned above with more space)
            current_action = self.actions[agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr]
            action_names = ['Idle', 'LL', 'UL',  'LR', 'UR' ]
            ap_x, ap_y = self.access_point_mapping(agent_x, agent_y, current_action)
            if current_action > 0 and (ap_x is None or ap_y is None):
                # Illegal action: show red box and label as Idle
                ax.text(agent_x, display_y + 0.4, 'Idle', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='white',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=illegal_action_color, alpha=0.8))
            else:
                ax.text(agent_x, display_y + 0.4, action_names[current_action], ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='black',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
            # State (positioned below with more space)
            agent_state = self.state[:, agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr]
            state_str = '[' + ', '.join([f'{s:.1f}' for s in agent_state.cpu().numpy()]) + ']'
            ax.text(agent_x, display_y - 0.4, state_str, ha='center', va='top',
                   fontsize=7, color='black',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        # Draw transmission arrows with better positioning (row-major, top-to-bottom)
        for agent_id in range(self.n_agents):
            agent_x = agent_id % self.grid_x
            agent_y = agent_id // self.grid_x
            display_y = self.grid_y - 1 - agent_y
            current_action = self.actions[agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr]
            agent_state = self.state[:, agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr]
            ap_x, ap_y = self.access_point_mapping(agent_x, agent_y, current_action)
            rewards = getattr(self, '_last_rewards', np.zeros(self.n_agents))
            if current_action > 0 and ap_x is not None and ap_y is not None:
                ap_display_y = self.grid_y - 2 - ap_y
                if rewards[agent_id] == 1:
                    arrow_color = transmission_color
                    arrow_width = 3
                else:
                    arrow_color = general_transmission_color
                    arrow_width = 2
                start_x = agent_x
                start_y = display_y
                end_x = ap_x + 0.5
                end_y = ap_display_y + 0.5
                dx = end_x - start_x
                dy = end_y - start_y
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    offset = 0.2
                    start_x = start_x + (dx/length) * offset
                    start_y = start_y + (dy/length) * offset
                arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                       arrowstyle='->', mutation_scale=20, 
                                       color=arrow_color, linewidth=arrow_width, alpha=0.8)
                ax.add_patch(arrow)

        # Convert to RGB array using savefig method
        if own_fig:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            data = np.array(img)[..., :3]  # Remove alpha channel if present
            plt.close(fig)
            return data
        else:
            # For interactive, render to the persistent axes and return the RGB array from the canvas
            self._mpl_fig.canvas.draw()
            width, height = self._mpl_fig.canvas.get_width_height()
            buf = self._mpl_fig.canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
            img = img[..., :3]  # Convert RGBA to RGB
            return img

    def render_realtime(self, show=True, save_path=None):
        """
        Render the current environment state in real-time.
        This can be called during training to see the current state immediately.
        
        Args:
            show: Whether to display the plot
            save_path: Optional path to save the current frame
        """
        if self.render_mode != "rgb_array":
            print("Warning: render_mode must be 'rgb_array' for real-time visualization")
            return None
            
        rgb_array = self._render_rgb_array()
        
        if show:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_array)
            plt.axis('off')
            plt.title(f'Wireless Communication Environment - Step {self.num_moves}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)  # Brief pause to show the frame
            plt.close()
        
        if save_path:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_array)
            plt.axis('off')
            plt.title(f'Wireless Communication Environment - Step {self.num_moves}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return rgb_array

    def start_frame_collection(self):
        """Start collecting frames for GIF creation."""
        self.frames = []
        self.collecting_frames = True
        print("Frame collection started. Call env.render() during training to collect frames.")

    def stop_frame_collection(self):
        """Stop collecting frames."""
        self.collecting_frames = False
        print(f"Frame collection stopped. Collected {len(self.frames)} frames.")

    def _save_gif(self, gif_path=None, fps=3, dpi=100):
        """
        Save collected frames as a GIF.
        
        Args:
            gif_path: Path to save the GIF (default: current working directory)
            fps: Frames per second for the GIF
            dpi: DPI for the GIF
        """
        if not self.frames:
            print("No frames collected. Call start_frame_collection() and env.render() first.")
            return None
        
        if gif_path is None:
            gif_path = os.path.join(os.getcwd(), 'wireless_comm_training.gif')
        
        print(f"Saving {len(self.frames)} frames as GIF to {gif_path}...")
        
        import matplotlib.pyplot as plt
        from matplotlib import animation
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        im = ax.imshow(self.frames[0])
        
        def update(i):
            im.set_data(self.frames[i])
            # ax.set_title(f'Wireless Communication Environment - Step {i + 1}', fontsize=14, fontweight='bold')
            return [im]
        
        anim = animation.FuncAnimation(fig, update, frames=len(self.frames), 
                                     interval=int(1000/fps), blit=True, repeat=True)
        anim.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
        plt.close(fig)
         
        print(f"GIF saved successfully!")
        return gif_path

    def close(self):
        """Close the environment. If save_gif is True, save the collected frames as a GIF."""
        if self.save_gif:
            if not self.frames:
                print("Warning: save_gif is True but no frames were collected. No GIF will be saved.")
            else:
                # print(f"Saving {len(self.frames)} frames as GIF to {self.gif_path}...")
                self._save_gif(self.gif_path)
        # Close the matplotlib window if open
        if self._mpl_fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._mpl_fig)
            self._mpl_fig = None
            self._mpl_ax = None
            self._mpl_im = None
        self.closed = True 