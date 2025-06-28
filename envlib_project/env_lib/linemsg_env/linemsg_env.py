"""
Line Message Environment - A standard gym environment for multi-agent line message passing.

This environment simulates a line of agents that must pass messages along the line
while dealing with message loss and random events.

Parameters:
- num_agents: Number of agents in the line (default: 10)
- n_obs_neighbors: Number of neighbors to observe (default: 1)
- max_iter: Maximum number of steps per episode (default: 50)
- render_mode: Rendering mode ('human', 'rgb_array', None)
"""

import numpy as np
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle, seeding


class LineMsgEnv(Env, EzPickle):
    """
    Line Message Environment - Standard gym environment for multi-agent line message passing.
    
    This environment simulates a line of agents that must pass messages along the line
    while dealing with message loss and random events.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "LineMsg-v0",
        "render_fps": 10,
    }

    def __init__(
        self,
        num_agents=10,
        n_obs_neighbors=1,
        max_iter=50,
        render_mode=None,
    ):
        """
        Initialize the Line Message environment.
        
        Args:
            num_agents: Number of agents in the line
            n_obs_neighbors: Number of neighbors to observe (minimum 1)
            max_iter: Maximum number of steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', None)
        """
        EzPickle.__init__(
            self,
            num_agents,
            n_obs_neighbors,
            max_iter,
            render_mode,
        )
        
        # Environment parameters
        self.num_agents = num_agents
        self.n_obs_nghbr = max(1, n_obs_neighbors)  # Ensure minimum neighborhood size of 1
        self.max_iter = max_iter
        self.render_mode = render_mode
        
        # Agent setup
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        
        # Action and observation spaces
        self.action_space = Discrete(2 ** self.num_agents)  # Each agent has 2 actions (0 or 1)
        
        # Observation space: each agent gets its local state
        obs_dim = self.n_obs_nghbr * 2 + 1
        self.observation_space = Box(
            low=0, 
            high=3, 
            shape=(self.num_agents, obs_dim), 
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

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed(seed)
            
        # Initialize state
        self.state = np.ones(self.num_agents + self.n_obs_nghbr * 2, dtype=int)
        self.state[:self.n_obs_nghbr] = 2
        self.state[-self.n_obs_nghbr:] = 2
        
        # Initialize actions
        self.actions = np.zeros(self.num_agents + self.n_obs_nghbr * 2, dtype=int) + 2
        
        self.num_moves = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Get observations for all agents."""
        if self.state is None:
            # Initialize state if not already done
            self.reset()
            
        assert self.state is not None  # Type assertion for linter
        obs = np.zeros((self.num_agents, self.n_obs_nghbr * 2 + 1))
        
        for i in range(self.num_agents):
            obs[i] = self.state[i:i + self.n_obs_nghbr * 2 + 1]
            
        return obs

    def step(self, action):
        """Take a step in the environment."""
        if self.state is None or self.actions is None:
            # Initialize state if not already done
            self.reset()
            
        assert self.state is not None and self.actions is not None  # Type assertion for linter
        
        # Convert discrete action to individual agent actions
        action_array = np.zeros(self.num_agents, dtype=int)
        for i in range(self.num_agents):
            action_array[i] = (action // (2 ** i)) % 2
        
        # Update actions
        self.actions[self.n_obs_nghbr:-self.n_obs_nghbr] = action_array
        
        # Update state based on message passing dynamics
        new_state_agent0 = self.state[self.n_obs_nghbr + 1]
        new_state_agent_last = self.actions[-self.n_obs_nghbr - 1]
        
        # Calculate new state for middle agents
        new_state = ((self.state[self.n_obs_nghbr + 2:-self.n_obs_nghbr] + 
                      self.actions[self.n_obs_nghbr + 1:-self.n_obs_nghbr-1]) == 2) * 1.0
        
        # Apply random events (message loss)
        ran_mask = ((1 - self.state[self.n_obs_nghbr + 2:-self.n_obs_nghbr]) + 
                    self.actions[self.n_obs_nghbr + 1:-self.n_obs_nghbr-1]) == 2
        random_prob = self.np_random.random(self.num_agents - 2)
        new_state[ran_mask & (random_prob < 0.8)] = 1
        
        # Update state
        self.state[self.n_obs_nghbr + 1:-self.n_obs_nghbr - 1] = new_state
        self.state[self.n_obs_nghbr] = new_state_agent0
        self.state[-self.n_obs_nghbr-1] = new_state_agent_last
        
        # Calculate rewards
        rewards_np = (self.state[self.n_obs_nghbr:-self.n_obs_nghbr] == 1) * 0.1
        rewards_np[0] = rewards_np[0] * 10  # First agent gets 10x reward
        
        # Update step counter
        self.num_moves += 1
        terminated = False
        truncated = self.num_moves >= self.max_iter
        
        # Calculate total reward
        total_reward = np.sum(rewards_np)
        
        return self._get_obs(), total_reward, terminated, truncated, {}

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return

        if self.state is None or self.actions is None:
            return "Environment not initialized. Call reset() first."

        if self.render_mode == "human":
            string = "Current state: \n"
            for i, agent in enumerate(self.agents):
                string += f"Agent {i}: action = {self.actions[i + self.n_obs_nghbr]}, " \
                          f"state = {self.state[i + self.n_obs_nghbr]}\n"
            print(string)
            return string
        elif self.render_mode == "rgb_array":
            # Create a simple visualization as an array
            # This is a placeholder - you could create a more sophisticated visualization
            vis_array = np.zeros((self.num_agents * 30, 100, 3), dtype=np.uint8)
            
            for i in range(self.num_agents):
                # Color based on current state
                if self.state[i + self.n_obs_nghbr] == 1:
                    color = [0, 255, 0]  # Green for message
                elif self.state[i + self.n_obs_nghbr] == 0:
                    color = [255, 0, 0]  # Red for no message
                else:
                    color = [128, 128, 128]  # Gray for unknown
                
                vis_array[i*30:(i+1)*30, :] = color
            
            return vis_array

    def close(self):
        """Close the environment."""
        if not self.closed:
            self.closed = True 