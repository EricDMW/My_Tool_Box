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
        
        # Action and observation spaces
        # Each agent has 5 actions (0-4), use MultiDiscrete for scalability
        self.action_space = MultiDiscrete([5] * self.n_agents)
        
        # Observation space: each agent gets its local state
        obs_dim = self.ddl * ((self.n_obs_nghbr * 2 + 1) ** 2)
        self.observation_space = Box(
            low=0, 
            high=2, 
            shape=(self.n_agents, obs_dim), 
            dtype=np.float32
        )
        
        # Game state
        self.state = None
        self.actions = None
        self.num_moves = 0
        
        self.closed = False
        self.seed()

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def access_point_mapping(self, i, j, agent_action):
        """Map agent action to access point coordinates."""
        if agent_action == 0:
            return None, None
        if agent_action == 1:  # transit to the up and left access point
            agent_access_point_x = i - 1
            agent_access_point_y = j - 1
        elif agent_action == 2:  # down and left
            agent_access_point_x = i
            agent_access_point_y = j - 1
        elif agent_action == 3:  # up and right
            agent_access_point_x = i - 1
            agent_access_point_y = j
        elif agent_action == 4:  # down and right
            agent_access_point_x = i
            agent_access_point_y = j
        else:
            raise ValueError(f'agent_action = {agent_action} is not defined!')
        
        if agent_access_point_x < 0 or agent_access_point_x >= self.grid_x - 1 or agent_access_point_y < 0 or \
                agent_access_point_y >= self.grid_y - 1:
            return None, None
        else:
            return agent_access_point_x, agent_access_point_y

    def check_transmission_fail(self, agent_access_point_x, agent_access_point_y, access_point_profile):
        """Check if transmission fails due to interference."""
        if agent_access_point_x is None or agent_access_point_y is None:
            return True
        return access_point_profile[agent_access_point_x, agent_access_point_y] != 1

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed(seed)
            
        # Initialize state
        self.state = np.full((self.ddl, self.grid_x + 2 * self.n_obs_nghbr, self.grid_y + 2 * self.n_obs_nghbr),
                             fill_value=2, dtype=np.float32)
        self.state[:, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = 0
        
        # Initialize actions
        self.actions = np.zeros((self.grid_x + 2 * self.n_obs_nghbr, self.grid_y + 2 * self.n_obs_nghbr)).astype(int)
        
        # Initialize packet arrivals
        self.state[:, self.n_obs_nghbr:self.grid_x + self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y + self.n_obs_nghbr] = \
            self.np_random.choice(2, size=(self.ddl, self.grid_x, self.grid_y))
        
        self.num_moves = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Get observations for all agents."""
        assert self.state is not None
        obs = np.zeros((self.n_agents, self.ddl * ((self.n_obs_nghbr * 2 + 1) ** 2)))
        
        for i in range(self.n_agents):
            agent_x = i // self.grid_y
            agent_y = i % self.grid_y
            obs[i] = self.state[:, 
                               agent_x: agent_x + self.n_obs_nghbr * 2 + 1,
                               agent_y: agent_y + self.n_obs_nghbr * 2 + 1].reshape(-1)
            
        return obs

    def step(self, action):
        """Take a step in the environment."""
        assert self.state is not None
        # Convert discrete action to individual agent actions
        action_array = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            action_array[i] = (action // (5 ** i)) % 5
        
        # Create access point profile
        access_point_profile = np.zeros((self.grid_x - 1, self.grid_y - 1))
        for agent_id in range(self.n_agents):
            agent_x = agent_id // self.grid_y
            agent_y = agent_id % self.grid_y
            agent_access_point_x, agent_access_point_y = self.access_point_mapping(
                agent_x, agent_y, action_array[agent_id]
            )
            if agent_access_point_x is not None and agent_access_point_y is not None:
                access_point_profile[agent_access_point_x, agent_access_point_y] += 1

        # Calculate rewards
        rewards = np.zeros(self.n_agents)
        
        # Update state and calculate rewards for each agent
        for agent_id in range(self.n_agents):
            i = agent_id // self.grid_y
            j = agent_id % self.grid_y
            agent_action = action_array[agent_id]
            agent_access_point_x, agent_access_point_y = self.access_point_mapping(i, j, agent_action)
            
            # Check if transmission conditions are met
            if not (agent_action == 0 or \
                    self.state[:, self.n_obs_nghbr + i, self.n_obs_nghbr + j].max() == 0 or \
                    (self.state[:, self.n_obs_nghbr + i, self.n_obs_nghbr + j].max() == 1 and \
                     self.check_transmission_fail(agent_access_point_x, agent_access_point_y, access_point_profile))):
                
                # Find the leftmost "1" in the agent's state
                idx = 0
                while self.state[idx, self.n_obs_nghbr + i, self.n_obs_nghbr + j] == 0:
                    idx += 1
                
                # Attempt transmission
                if self.np_random.random() <= self.q:
                    self.state[idx, self.n_obs_nghbr + i, self.n_obs_nghbr + j] = 0
                    rewards[agent_id] = 1

        # Update state: left shift and append new packets
        self.state[:-1, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = \
            copy.deepcopy(self.state[1:, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr])
        self.state[-1, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = \
            self.np_random.random((self.grid_x, self.grid_y)) <= self.p

        # Update step counter
        self.num_moves += 1
        terminated = False
        truncated = self.num_moves >= self.max_iter
        
        # Calculate total reward
        total_reward = np.sum(rewards)
        
        return self._get_obs(), total_reward, terminated, truncated, {}

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return

        assert self.state is not None and self.actions is not None

        if self.render_mode == "human":
            string = "Current state: \n"
            for i, agent in enumerate(self.agents):
                agent_x = i // self.grid_y
                agent_y = i % self.grid_y
                string += f"Agent {i}: action = {self.actions[agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr]}, " \
                          f"state = {self.state[:, agent_x + self.n_obs_nghbr, agent_y + self.n_obs_nghbr]}\n"
            print(string)
            return string
        elif self.render_mode == "rgb_array":
            # Create a simple visualization as an array
            # This is a placeholder - you could create a more sophisticated visualization
            vis_array = np.zeros((self.grid_x * 50, self.grid_y * 50, 3), dtype=np.uint8)
            
            for i in range(self.grid_x):
                for j in range(self.grid_y):
                    agent_id = i * self.grid_y + j
                    # Color based on current state
                    if self.state[0, i + self.n_obs_nghbr, j + self.n_obs_nghbr] == 1:
                        color = [255, 0, 0]  # Red for packet
                    else:
                        color = [0, 255, 0]  # Green for no packet
                    
                    vis_array[i*50:(i+1)*50, j*50:(j+1)*50] = color
            
            return vis_array

    def close(self):
        """Close the environment."""
        if not self.closed:
            self.closed = True 