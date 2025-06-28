import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List, Union
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle
import math


class KuramotoOscillatorEnvTorch(gym.Env):
    """
    PyTorch-based Kuramoto Oscillator Synchronization Environment
    
    This environment simulates multiple systems of N coupled oscillators following the Kuramoto model.
    All computations are performed on the specified device (CPU/GPU) using PyTorch tensors.
    
    The Kuramoto model is described by:
    dθᵢ/dt = ωᵢ + (K/N) * Σⱼ sin(θⱼ - θᵢ)
    
    Features:
    - Multi-agent support (batch processing)
    - Device-aware (CPU/GPU)
    - All parameters controllable
    - PyTorch tensor operations throughout
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, 
                 n_oscillators: int = 10,
                 n_agents: int = 1,
                 dt: float = 0.01,
                 max_steps: int = 1000,
                 coupling_range: Tuple[float, float] = (0.0, 5.0),
                 control_input_range: Tuple[float, float] = (-1.0, 1.0),
                 natural_freq_range: Tuple[float, float] = (0.5, 2.0),
                 device: str = "cpu",
                 render_mode: Optional[str] = None,
                 integration_method: str = "euler",
                 reward_type: str = "order_parameter",
                 noise_std: float = 0.0,
                 topology: str = "fully_connected",
                 adj_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 target_frequency: float = 1.0,
                 coupling_mode: str = "dynamic",
                 constant_coupling_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """
        Initialize the PyTorch-based Kuramoto oscillator environment.
        
        Args:
            n_oscillators: Number of oscillators per system
            n_agents: Number of parallel agents/systems
            dt: Time step for integration
            max_steps: Maximum number of steps per episode
            coupling_range: Range for coupling strength values (used only in dynamic mode)
            control_input_range: Range for external control input values
            natural_freq_range: Range for natural frequencies
            device: Device to run computations on ('cpu' or 'cuda')
            render_mode: Rendering mode ('human' or 'rgb_array')
            integration_method: Integration method ('euler', 'rk4')
            reward_type: Reward type ('order_parameter', 'phase_coherence', 'combined', 'frequency_synchronization')
            noise_std: Standard deviation of noise added to dynamics
            topology: Network topology ('fully_connected', 'ring', 'star', 'random', 'custom')
            adj_matrix: An adjacency matrix to define a custom topology.
            target_frequency: The target frequency for the 'frequency_synchronization' reward.
            coupling_mode: Either 'dynamic' (coupling strengths controlled by actions) or 'constant' (fixed coupling matrix)
            constant_coupling_matrix: Fixed coupling matrix to use when coupling_mode='constant'
        """
        super().__init__()
        
        # Environment parameters
        self.n_oscillators = n_oscillators
        self.n_agents = n_agents
        self.dt = dt
        self.max_steps = max_steps
        self.coupling_range = coupling_range
        self.control_input_range = control_input_range
        self.natural_freq_range = natural_freq_range
        self.device = torch.device(device)
        self.render_mode = render_mode
        self.integration_method = integration_method
        self.reward_type = reward_type
        self.noise_std = noise_std
        self.topology = topology
        self.adj_matrix = adj_matrix
        self.target_frequency = target_frequency
        self.coupling_mode = coupling_mode
        self.constant_coupling_matrix = constant_coupling_matrix
        
        # Current state (all as PyTorch tensors)
        self.phases: Optional[torch.Tensor] = None  # Shape: (n_agents, n_oscillators)
        self.natural_frequencies: Optional[torch.Tensor] = None  # Shape: (n_agents, n_oscillators)
        self.coupling_strengths: Optional[torch.Tensor] = None  # Shape: (n_agents, n_couplings) - only used in dynamic mode
        self.control_inputs: Optional[torch.Tensor] = None # Shape: (n_agents, n_oscillators)
        self.step_count = 0
        
        # Initialize topology matrix
        self._init_topology()
        
        # Initialize coupling matrix based on mode
        self._init_coupling_matrix()
        
        # Define action space based on coupling mode
        if self.coupling_mode == "dynamic":
            # Action space: control inputs and coupling strengths
            self.n_couplings = self._get_n_couplings()
            
            lows = np.concatenate([
                np.full(self.n_oscillators, self.control_input_range[0]),
                np.full(self.n_couplings, self.coupling_range[0])
            ]).astype(np.float32)
            highs = np.concatenate([
                np.full(self.n_oscillators, self.control_input_range[1]),
                np.full(self.n_couplings, self.coupling_range[1])
            ]).astype(np.float32)
            self.action_space = spaces.Box(
                low=lows,
                high=highs,
                dtype=np.float32
            )
        else:  # constant mode
            # Action space: only control inputs
            self.action_space = spaces.Box(
                low=np.full(self.n_oscillators, self.control_input_range[0]),
                high=np.full(self.n_oscillators, self.control_input_range[1]),
                dtype=np.float32
            )
        
        # Define observation space: phases, freqs, couplings (if dynamic), control_inputs
        if self.coupling_mode == "dynamic":
            n_obs_dim = self.n_oscillators * 3 + self.n_couplings
        else:
            n_obs_dim = self.n_oscillators * 3  # No coupling strengths in observation for constant mode
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_obs_dim,),
            dtype=np.float32
        )
        
        # Initialize coupling matrix indices for efficient access (only needed for dynamic mode)
        if self.coupling_mode == "dynamic":
            self._init_coupling_indices()
        
        # Rendering setup
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self.phase_history = []
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(42)
    
    def _init_topology(self):
        """Initialize network topology matrix."""
        if self.adj_matrix is not None:
            if isinstance(self.adj_matrix, np.ndarray):
                self.topology_matrix = torch.from_numpy(self.adj_matrix).float().to(self.device)
            else:
                self.topology_matrix = self.adj_matrix.float().to(self.device)
            
            # Validation
            assert self.topology_matrix.dim() == 2, "Adjacency matrix must be 2D"
            assert self.topology_matrix.shape[0] == self.topology_matrix.shape[1], "Adjacency matrix must be square"
            assert self.topology_matrix.shape[0] == self.n_oscillators, "Adjacency matrix size must match n_oscillators"
            # assert torch.all(self.topology_matrix == self.topology_matrix.t()), "Adjacency matrix must be symmetric"
            
            self.topology = "custom"
        elif self.topology == "fully_connected":
            # All oscillators connected to each other
            self.topology_matrix = torch.ones(self.n_oscillators, self.n_oscillators, device=self.device)
            torch.diagonal(self.topology_matrix)[:] = 0  # No self-connections
            
        elif self.topology == "ring":
            # Ring topology: each oscillator connected to neighbors
            self.topology_matrix = torch.zeros(self.n_oscillators, self.n_oscillators, device=self.device)
            for i in range(self.n_oscillators):
                self.topology_matrix[i, (i-1) % self.n_oscillators] = 1
                self.topology_matrix[i, (i+1) % self.n_oscillators] = 1
                
        elif self.topology == "star":
            # Star topology: central oscillator connected to all others
            self.topology_matrix = torch.zeros(self.n_oscillators, self.n_oscillators, device=self.device)
            self.topology_matrix[0, 1:] = 1  # Central oscillator
            self.topology_matrix[1:, 0] = 1  # Others connected to center
            
        elif self.topology == "random":
            # Random topology with 50% connection probability
            self.topology_matrix = torch.rand(self.n_oscillators, self.n_oscillators, device=self.device) > 0.5
            torch.diagonal(self.topology_matrix)[:] = 0  # No self-connections
            # Ensure symmetry
            self.topology_matrix = (self.topology_matrix + self.topology_matrix.t()) > 0
            
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
    
    def _get_n_couplings(self):
        """Get number of unique coupling strengths based on topology."""
        if self.topology == "fully_connected":
            return self.n_oscillators * (self.n_oscillators - 1) // 2
        else:
            # For other topologies, count actual connections
            return int(torch.sum(self.topology_matrix) // 2)
    
    def _init_coupling_indices(self):
        """Initialize indices for mapping coupling strengths to coupling matrix."""
        self.coupling_indices = []
        if self.topology == "fully_connected":
            for i in range(self.n_oscillators):
                for j in range(i + 1, self.n_oscillators):
                    self.coupling_indices.append((i, j))
        else:
            # For other topologies, find actual connections
            connections = torch.nonzero(self.topology_matrix)
            for i, j in connections:
                if i < j:  # Avoid duplicates
                    self.coupling_indices.append((i.item(), j.item()))
    
    def _coupling_matrix_from_actions(self, actions):
        """Convert action tensor to symmetric coupling matrix."""
        # actions shape: (n_agents, n_couplings)
        batch_size = actions.shape[0]
        coupling_matrix = torch.zeros(batch_size, self.n_oscillators, self.n_oscillators, device=self.device)
        
        for idx, (i, j) in enumerate(self.coupling_indices):
            coupling_matrix[:, i, j] = actions[:, idx]
            coupling_matrix[:, j, i] = actions[:, idx]  # Symmetric
            
        return coupling_matrix
    
    def _compute_order_parameter(self, phases):
        """Compute the Kuramoto order parameter (measure of synchronization)."""
        # phases shape: (n_agents, n_oscillators)
        complex_phases = torch.exp(1j * phases)
        order_param = torch.abs(torch.mean(complex_phases, dim=1))
        return order_param
    
    def _compute_phase_coherence(self, phases):
        """Compute phase coherence (alternative synchronization measure)."""
        # Normalize phases to [0, 2π]
        phases_norm = phases % (2 * torch.pi)
        # Compute variance of phases
        phase_variance = torch.var(phases_norm, dim=1)
        # Coherence is inversely related to variance
        coherence = torch.exp(-phase_variance)
        return coherence
    
    def _kuramoto_dynamics(self, phases, natural_frequencies, coupling_matrix, control_inputs):
        """Compute the Kuramoto dynamics for all oscillators using PyTorch."""
        # All inputs are PyTorch tensors
        # phases: (n_agents, n_oscillators)
        # natural_frequencies: (n_agents, n_oscillators)
        # coupling_matrix: (n_agents, n_oscillators, n_oscillators)
        
        # Natural frequency term and control input
        dphases_dt = natural_frequencies + control_inputs
        
        # Coupling term
        # Compute sin(θⱼ - θᵢ) for all pairs
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)  # (n_agents, n_oscillators, n_oscillators)
        sin_diff = torch.sin(phase_diff)
        
        # Apply coupling matrix and sum
        coupling_effect = coupling_matrix * sin_diff
        coupling_sum = torch.sum(coupling_effect, dim=2)  # Sum over j
        
        dphases_dt += coupling_sum / self.n_oscillators
            
        return dphases_dt
    
    def _integrate_rk4(self, phases, natural_frequencies, coupling_matrix, control_inputs):
        """Fourth-order Runge-Kutta integration."""
        dt = self.dt
        
        k1 = self._kuramoto_dynamics(phases, natural_frequencies, coupling_matrix, control_inputs)
        k2 = self._kuramoto_dynamics(phases + 0.5 * dt * k1, natural_frequencies, coupling_matrix, control_inputs)
        k3 = self._kuramoto_dynamics(phases + 0.5 * dt * k2, natural_frequencies, coupling_matrix, control_inputs)
        k4 = self._kuramoto_dynamics(phases + dt * k3, natural_frequencies, coupling_matrix, control_inputs)
        
        return phases + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), k1
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(seed)
        
        # Initialize phases randomly in [-π, π]
        self.phases = (torch.rand(self.n_agents, self.n_oscillators, device=self.device) * 2 - 1) * torch.pi
        
        # Initialize natural frequencies
        self.natural_frequencies = torch.rand(self.n_agents, self.n_oscillators, device=self.device) * \
                                  (self.natural_freq_range[1] - self.natural_freq_range[0]) + \
                                  self.natural_freq_range[0]
        
        # Initialize coupling strengths (only in dynamic mode)
        if self.coupling_mode == "dynamic":
            self.coupling_strengths = torch.rand(self.n_agents, self.n_couplings, device=self.device) * \
                                     (self.coupling_range[1] - self.coupling_range[0]) + \
                                     self.coupling_range[0]
        else:
            self.coupling_strengths = None
        
        # Initialize control inputs to zero
        self.control_inputs = torch.zeros(self.n_agents, self.n_oscillators, device=self.device)
        
        self.step_count = 0
        
        # Create observation (for first agent only, as per Gymnasium convention)
        observation = self._create_observation(0)
        
        # Initialize phase history after phases are set
        assert self.phases is not None
        self.phase_history = [self.phases.clone().cpu().numpy()]
        
        assert self.phases is not None and self.natural_frequencies is not None
        info = {
            'order_parameter': self._compute_order_parameter(self.phases).cpu().numpy(),
            'phase_coherence': self._compute_phase_coherence(self.phases).cpu().numpy(),
            'natural_frequencies': self.natural_frequencies.cpu().numpy(),
            'phases': self.phases.cpu().numpy(),
            'device': str(self.device),
            'n_agents': self.n_agents
        }
        
        return observation.astype(np.float32), info
    
    def _create_observation(self, agent_idx=0):
        """Create observation for a specific agent."""
        assert self.phases is not None and \
               self.natural_frequencies is not None and \
               self.control_inputs is not None
        
        if self.coupling_mode == "dynamic":
            assert self.coupling_strengths is not None
            return torch.cat([
                self.phases[agent_idx],
                self.natural_frequencies[agent_idx],
                self.coupling_strengths[agent_idx],
                self.control_inputs[agent_idx]
            ]).cpu().numpy()
        else:  # constant mode
            return torch.cat([
                self.phases[agent_idx],
                self.natural_frequencies[agent_idx],
                self.control_inputs[agent_idx]
            ]).cpu().numpy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        assert self.phases is not None and self.natural_frequencies is not None and self.control_inputs is not None
        
        # Convert action to tensor and expand for all agents
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0).expand(self.n_agents, -1)
        
        if self.coupling_mode == "dynamic":
            # Split action into control inputs and coupling strengths
            control_actions = action_tensor[:, :self.n_oscillators]
            coupling_actions = action_tensor[:, self.n_oscillators:]
            
            # Update state
            self.control_inputs = torch.clamp(control_actions, self.control_input_range[0], self.control_input_range[1])
            self.coupling_strengths = torch.clamp(coupling_actions, self.coupling_range[0], self.coupling_range[1])
            
            # Convert actions to coupling matrix
            coupling_matrix = self._coupling_matrix_from_actions(self.coupling_strengths)
        else:  # constant mode
            # Only control inputs in action
            control_actions = action_tensor
            
            # Update state
            self.control_inputs = torch.clamp(control_actions, self.control_input_range[0], self.control_input_range[1])
            
            # Use constant coupling matrix
            coupling_matrix = self.coupling_matrix
        
        dphases_dt = None
        # Integrate Kuramoto dynamics
        if self.integration_method == "euler":
            dphases_dt = self._kuramoto_dynamics(self.phases, self.natural_frequencies, coupling_matrix, self.control_inputs)
            self.phases += dphases_dt * self.dt
        elif self.integration_method == "rk4":
            self.phases, dphases_dt = self._integrate_rk4(self.phases, self.natural_frequencies, coupling_matrix, self.control_inputs)
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
        
        assert self.phases is not None
        # Add noise if specified (as per the equation)
        if self.noise_std > 0:
            noise = torch.randn_like(self.phases) * self.noise_std
            self.phases += noise
            
        # Normalize phases to [-π, π]
        self.phases = (self.phases + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Store phase history for rendering
        assert self.phases is not None
        self.phase_history.append(self.phases.clone().cpu().numpy())
        
        # Compute synchronization measures
        order_parameter = self._compute_order_parameter(self.phases)
        phase_coherence = self._compute_phase_coherence(self.phases)
        
        # Create observation (for first agent)
        observation = self._create_observation(0)
        
        # Compute reward based on synchronization
        reward = 0.0
        if self.reward_type == "frequency_synchronization":
            assert dphases_dt is not None
            agent_rewards = -torch.abs(dphases_dt - self.target_frequency)
            reward = torch.mean(agent_rewards[0]).item()
        elif self.reward_type == "order_parameter":
            reward = order_parameter[0].item()
        elif self.reward_type == "phase_coherence":
            reward = phase_coherence[0].item()
        elif self.reward_type == "combined":
            reward = order_parameter[0].item() + phase_coherence[0].item()
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
        # Check if episode is done
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # Additional termination condition: perfect synchronization
        if order_parameter[0].item() > 0.99:
            reward += 10.0  # Bonus for achieving synchronization
            done = True
        
        assert self.natural_frequencies is not None and self.phases is not None
        info = {
            'order_parameter': order_parameter.cpu().numpy(),
            'phase_coherence': phase_coherence.cpu().numpy(),
            'step_count': self.step_count,
            'natural_frequencies': self.natural_frequencies.cpu().numpy(),
            'phases': self.phases.cpu().numpy(),
            'coupling_matrix': coupling_matrix.cpu().numpy() if coupling_matrix is not None else None,
            'device': str(self.device),
            'n_agents': self.n_agents
        }

        if dphases_dt is not None:
            info['dphases_dt'] = dphases_dt.cpu().numpy()
        
        return observation.astype(np.float32), reward, done, False, info
    
    def get_batch_observations(self) -> torch.Tensor:
        """Get observations for all agents as a batch tensor."""
        assert self.phases is not None and \
               self.natural_frequencies is not None and \
               self.control_inputs is not None
        
        observations = []
        for i in range(self.n_agents):
            if self.coupling_mode == "dynamic":
                assert self.coupling_strengths is not None
                obs = torch.cat([
                    self.phases[i],
                    self.natural_frequencies[i],
                    self.coupling_strengths[i],
                    self.control_inputs[i]
                ])
            else:  # constant mode
                obs = torch.cat([
                    self.phases[i],
                    self.natural_frequencies[i],
                    self.control_inputs[i]
                ])
            observations.append(obs)
        return torch.stack(observations)
    
    def get_batch_rewards(self, dphases_dt: torch.Tensor) -> torch.Tensor:
        """
        Get rewards for all agents as a batch tensor.
        The reward depends on the instantaneous frequency (dphases_dt), which is action-dependent.
        """
        if self.reward_type == "frequency_synchronization":
            agent_rewards = -torch.abs(dphases_dt - self.target_frequency)
            return torch.mean(agent_rewards, dim=1) # Mean over oscillators for each agent
        else:
            assert self.phases is not None
            order_parameter = self._compute_order_parameter(self.phases)
            phase_coherence = self._compute_phase_coherence(self.phases)
            
            if self.reward_type == "order_parameter":
                return order_parameter
            elif self.reward_type == "phase_coherence":
                return phase_coherence
            elif self.reward_type == "combined":
                return order_parameter + phase_coherence
            else:
                raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def render(self):
        """Render the current state of the oscillators."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        """Render a single frame showing oscillator phases."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        assert self.ax is not None
        self.ax.clear()
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw unit circle
        circle = Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
        self.ax.add_patch(circle)
        
        # Plot oscillators on unit circle (for first agent)
        assert self.phases is not None and self.natural_frequencies is not None
        phases = self.phases[0].cpu().numpy()
        natural_freqs = self.natural_frequencies[0].cpu().numpy()
        
        x_positions = np.cos(phases)
        y_positions = np.sin(phases)
        
        # Color oscillators based on their natural frequency
        colors = cm.get_cmap('viridis')((natural_freqs - self.natural_freq_range[0]) /
                               (self.natural_freq_range[1] - self.natural_freq_range[0]))
        
        scatter = self.ax.scatter(x_positions, y_positions, c=colors, s=100, alpha=0.7)
        
        # Add oscillator labels
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            self.ax.annotate(f'O{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Show order parameter
        order_param = self._compute_order_parameter(self.phases)[0].item()
        self.ax.set_title(f'Kuramoto Oscillators (PyTorch) - Order Parameter: {order_param:.3f}')
        
        if self.render_mode == "rgb_array":
            assert self.fig is not None
            self.fig.canvas.draw()  # type: ignore
            buf = self.fig.canvas.buffer_rgba()  # type: ignore
            img = np.asarray(buf)
            return img
    
    def close(self):
        """Close the environment and cleanup."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None 

    def _init_coupling_matrix(self):
        """Initialize coupling matrix based on coupling mode."""
        if self.coupling_mode == "constant":
            if self.constant_coupling_matrix is None:
                raise ValueError("constant_coupling_matrix must be provided when coupling_mode='constant'")
            
            # Convert to tensor and validate
            if isinstance(self.constant_coupling_matrix, np.ndarray):
                self.coupling_matrix = torch.from_numpy(self.constant_coupling_matrix).float().to(self.device)
            else:
                self.coupling_matrix = self.constant_coupling_matrix.float().to(self.device)
            
            # Validation
            assert self.coupling_matrix.dim() == 2, "Constant coupling matrix must be 2D"
            assert self.coupling_matrix.shape[0] == self.coupling_matrix.shape[1], "Constant coupling matrix must be square"
            assert self.coupling_matrix.shape[0] == self.n_oscillators, "Constant coupling matrix size must match n_oscillators"
            
            # Expand for all agents
            self.coupling_matrix = self.coupling_matrix.unsqueeze(0).expand(self.n_agents, -1, -1)
            
        else:  # dynamic mode
            # Coupling matrix will be computed from actions in each step
            self.coupling_matrix = None 