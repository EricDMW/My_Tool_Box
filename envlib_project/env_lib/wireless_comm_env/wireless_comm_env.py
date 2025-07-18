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

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def access_point_mapping(self, x, y, agent_action):
        """Map agent action to access point coordinates (UL, LL, UR, LR) for (x, y) with y increasing upwards."""
        if agent_action == 0:
            return None, None
        # UL: up and left
        if agent_action == 1:
            ap_x, ap_y = x - 1, y
        # LL: left
        elif agent_action == 2:
            ap_x, ap_y = x - 1, y - 1
        # UR: up
        elif agent_action == 3:
            ap_x, ap_y = x, y
        # LR: right
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
        self.state = np.full((self.ddl, self.grid_x + 2 * self.n_obs_nghbr, self.grid_y + 2 * self.n_obs_nghbr),
                             fill_value=2, dtype=np.float32)
        self.state[:, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = 0
        
        # Initialize actions
        self.actions = np.zeros((self.grid_x + 2 * self.n_obs_nghbr, self.grid_y + 2 * self.n_obs_nghbr)).astype(int)
        
        # Initialize packet arrivals - matching original probability
        self.state[:, self.n_obs_nghbr:self.grid_x + self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y + self.n_obs_nghbr] = \
            self.np_random.choice(2, size=(self.ddl, self.grid_x, self.grid_y))
        
        self.num_moves = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Get observations for all agents (row-major order)."""
        assert self.state is not None
        obs = np.zeros((self.n_agents, self.ddl * ((self.n_obs_nghbr * 2 + 1) ** 2)))
        
        for i in range(self.n_agents):
            agent_x = i % self.grid_x
            agent_y = i // self.grid_x
            obs[i] = self.state[:, 
                               agent_x: agent_x + self.n_obs_nghbr * 2 + 1,
                               agent_y: agent_y + self.n_obs_nghbr * 2 + 1].reshape(-1)
        
        return obs

    def step(self, action):
        """Take a step in the environment (row-major order, correct conflict logic)."""
        assert self.state is not None
        action_array = np.asarray(action, dtype=int).flatten()
        self.actions = np.zeros((self.grid_x + 2 * self.n_obs_nghbr, self.grid_y + 2 * self.n_obs_nghbr)).astype(int)
        for agent_id in range(self.n_agents):
            agent_x = agent_id % self.grid_x
            agent_y = agent_id // self.grid_x
            self.actions[agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr] = action_array[agent_id]

        # Build mapping from access point to list of agents transmitting to it
        ap_to_agents = {}
        agent_ap = [None] * self.n_agents
        for agent_id in range(self.n_agents):
            agent_x = agent_id % self.grid_x
            agent_y = agent_id // self.grid_x
            agent_action = action_array[agent_id]
            ap_x, ap_y = self.access_point_mapping(agent_x, agent_y, agent_action)
            if agent_action > 0 and ap_x is not None and ap_y is not None:
                ap_key = (ap_x, ap_y)
                if ap_key not in ap_to_agents:
                    ap_to_agents[ap_key] = []
                ap_to_agents[ap_key].append(agent_id)
                agent_ap[agent_id] = ap_key
            else:
                agent_ap[agent_id] = None

        rewards = np.zeros(self.n_agents)
        # Update state and calculate rewards for each agent
        for agent_id in range(self.n_agents):
            agent_x = agent_id % self.grid_x
            agent_y = agent_id // self.grid_x
            agent_action = action_array[agent_id]
            ap_key = agent_ap[agent_id]
            # Only one agent can succeed per access point
            if agent_action > 0 and ap_key is not None and len(ap_to_agents[ap_key]) == 1:
                # Check if agent has a packet to send
                if self.state[:, agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr].max() > 0:
                    # Find the leftmost "1" in the agent's state
                    idx = 0
                    while self.state[idx, agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr] == 0:
                        idx += 1
                    # Attempt transmission (with success probability)
                    if self.np_random.random() <= self.q:
                        self.state[idx, agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr] = 0
                        rewards[agent_id] = 1
        # Update state: left shift and append new packets (matching original logic)
        self.state[:-1, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = \
            copy.deepcopy(self.state[1:, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr])
        self.state[-1, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = \
            self.np_random.random((self.grid_x, self.grid_y)) <= self.p
        self.num_moves += 1
        terminated = False
        truncated = self.num_moves >= self.max_iter
        total_reward = np.sum(rewards)
        return self._get_obs(), total_reward, terminated, truncated, {}

    def render(self):
        """
        Render the environment. If frame collection is active, also collect the frame.
        """
        if self.render_mode == "human":
            string = "Current state: \n"
            for i, agent in enumerate(self.agents):
                agent_x = i % self.grid_x
                agent_y = i // self.grid_x
                string += f"Agent {i}: action = {self.actions[agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr]}, " \
                          f"state = {self.state[:, agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr]}\n"
            print(string)
            return string
        elif self.render_mode == "rgb_array":
            rgb_array = self._render_rgb_array()
            
            # Collect frame if collection is active
            if self.collecting_frames:
                self.frames.append(rgb_array)
            
            return rgb_array

    def _render_rgb_array(self):
        """Create a detailed RGB array visualization of the environment."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyArrowPatch
        import numpy as np
        
        # Create figure with explicit backend
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xlim(-0.5, self.grid_x)
        ax.set_ylim(-0.5, self.grid_y)
        ax.set_aspect('equal')
        ax.axis('off')
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
            action_names = ['Idle', 'LL', 'UL',  'LR', 'UR']
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
            state_str = '[' + ', '.join([f'{s:.1f}' for s in agent_state]) + ']'
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
            if current_action > 0 and ap_x is not None and ap_y is not None:
                ap_display_y = self.grid_y - 2 - ap_y
                has_packet = np.any(agent_state == 1)
                # Determine arrow color
                if has_packet and np.random.random() < 0.3:
                    arrow_color = transmission_color
                    arrow_width = 3
                else:
                    arrow_color = general_transmission_color
                    arrow_width = 2
                # Adjust arrow start and end points - start from agent, point to access point
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
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        data = np.array(img)[..., :3]  # Remove alpha channel if present
        
        plt.close(fig)
        return data

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
            plt.title(f'Wireless Communication Environment - Step {self.num_moves}')
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)  # Brief pause to show the frame
            plt.close()
        
        if save_path:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_array)
            plt.axis('off')
            plt.title(f'Wireless Communication Environment - Step {self.num_moves}')
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

    def save_gif(self, gif_path=None, fps=3, dpi=100):
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
            ax.set_title(f'Wireless Communication Environment - Step {i + 1}', fontsize=14, fontweight='bold')
            return [im]
        
        anim = animation.FuncAnimation(fig, update, frames=len(self.frames), 
                                     interval=int(1000/fps), blit=True, repeat=True)
        anim.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
        plt.close(fig)
        
        print(f"GIF saved successfully!")
        return gif_path

    def close(self):
        """Close the environment."""
        if not self.closed:
            self.closed = True 