import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle
import math


class KuramotoOscillatorEnv(gym.Env):
    """
    Kuramoto Oscillator Synchronization Environment
    
    This environment simulates a system of N coupled oscillators following the Kuramoto model.
    The goal is to achieve synchronization by controlling the coupling strengths between oscillators.
    
    The Kuramoto model is described by:
    dθᵢ/dt = ωᵢ + (K/N) * Σⱼ sin(θⱼ - θᵢ)
    θ_i(t+1) = θ_i(t) + dt(ωᵢ + a_i(t) + (Σⱼ sin(θⱼ - θᵢ)) + epsilon_i(t))
    
    where:
    - θᵢ is the phase of oscillator i
    - ωᵢ is the natural frequency of oscillator i
    - K is the coupling strength
    - N is the number of oscillators
    - a_i(t) is the control input for oscillator i at time t
    - epsilon_i(t) is the noise for oscillator i at time t
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
                 adj_matrix: Optional[np.ndarray] = None,
                 target_frequency: float = 1.0,
                 coupling_mode: str = "dynamic",
                 constant_coupling_matrix: Optional[np.ndarray] = None):
        """
        Initialize the Kuramoto oscillator environment.
        
        Args:
            n_oscillators: Number of oscillators in the system
            n_agents: Number of parallel agents/systems (default: 1 for single agent)
            dt: Time step for integration
            max_steps: Maximum number of steps per episode
            coupling_range: Range for coupling strength values (used only in dynamic mode)
            control_input_range: Range for external control input values
            natural_freq_range: Range for natural frequencies
            device: Device parameter (ignored in numpy version, kept for compatibility)
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
        self.device = device  # Ignored in numpy version but kept for compatibility
        self.render_mode = render_mode
        self.integration_method = integration_method
        self.reward_type = reward_type
        self.noise_std = noise_std
        self.topology = topology
        self.adj_matrix = adj_matrix
        self.target_frequency = target_frequency
        self.coupling_mode = coupling_mode
        self.constant_coupling_matrix = constant_coupling_matrix
        
        # Current state
        self.phases: Optional[np.ndarray] = None
        self.natural_frequencies: Optional[np.ndarray] = None
        self.coupling_strengths: Optional[np.ndarray] = None
        self.control_inputs: Optional[np.ndarray] = None
        self.step_count = 0

        # Initialize topology
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

    def _init_topology(self):
        """Initialize network topology matrix."""
        if self.adj_matrix is not None:
            self.topology_matrix = self.adj_matrix
            # Validation
            assert self.topology_matrix.ndim == 2, "Adjacency matrix must be 2D"
            assert self.topology_matrix.shape[0] == self.topology_matrix.shape[1], "Adjacency matrix must be square"
            assert self.topology_matrix.shape[0] == self.n_oscillators, "Adjacency matrix size must match n_oscillators"
            self.topology = "custom"
        elif self.topology == "fully_connected":
            # All oscillators connected to each other
            self.topology_matrix = np.ones((self.n_oscillators, self.n_oscillators))
            np.fill_diagonal(self.topology_matrix, 0)  # No self-connections
        elif self.topology == "ring":
            # Ring topology: each oscillator connected to neighbors
            self.topology_matrix = np.zeros((self.n_oscillators, self.n_oscillators))
            for i in range(self.n_oscillators):
                self.topology_matrix[i, (i-1) % self.n_oscillators] = 1
                self.topology_matrix[i, (i+1) % self.n_oscillators] = 1
        elif self.topology == "star":
            # Star topology: central oscillator connected to all others
            self.topology_matrix = np.zeros((self.n_oscillators, self.n_oscillators))
            self.topology_matrix[0, 1:] = 1  # Central oscillator
            self.topology_matrix[1:, 0] = 1  # Others connected to center
        elif self.topology == "random":
            # Random topology with 50% connection probability
            np.random.seed(42)  # For reproducibility
            self.topology_matrix = (np.random.random((self.n_oscillators, self.n_oscillators)) > 0.5).astype(float)
            np.fill_diagonal(self.topology_matrix, 0)  # No self-connections
            # Ensure symmetry
            self.topology_matrix = ((self.topology_matrix + self.topology_matrix.T) > 0).astype(float)
        else:
            raise ValueError(f"Unknown topology: {self.topology}. Must be one of: fully_connected, ring, star, random, custom")
    
    def _init_coupling_matrix(self):
        """Initialize coupling matrix based on coupling mode."""
        if self.coupling_mode == "constant":
            if self.constant_coupling_matrix is None:
                raise ValueError("constant_coupling_matrix must be provided when coupling_mode='constant'")
            
            # Validate constant coupling matrix
            assert self.constant_coupling_matrix.ndim == 2, "Constant coupling matrix must be 2D"
            assert self.constant_coupling_matrix.shape[0] == self.constant_coupling_matrix.shape[1], "Constant coupling matrix must be square"
            assert self.constant_coupling_matrix.shape[0] == self.n_oscillators, "Constant coupling matrix size must match n_oscillators"
            
            self.coupling_matrix = self.constant_coupling_matrix.copy()
            
        else:  # dynamic mode
            # Coupling matrix will be computed from actions in each step
            self.coupling_matrix = None
    
    def _get_n_couplings(self):
        """Get number of unique coupling strengths based on topology."""
        return int(np.sum(self.topology_matrix) // 2)
        
    def _init_coupling_indices(self):
        """Initialize indices for mapping coupling strengths to coupling matrix."""
        self.coupling_indices = []
        rows, cols = np.where(np.triu(self.topology_matrix, 1))
        self.coupling_indices = list(zip(rows, cols))
                
    def _coupling_matrix_from_actions(self, actions):
        """Convert action vector to symmetric coupling matrix."""
        coupling_matrix = np.zeros((self.n_oscillators, self.n_oscillators))
        for idx, (i, j) in enumerate(self.coupling_indices):
            coupling_matrix[i, j] = actions[idx]
            coupling_matrix[j, i] = actions[idx]  # Symmetric
        return coupling_matrix
    
    def _compute_order_parameter(self, phases):
        """Compute the Kuramoto order parameter (measure of synchronization)."""
        complex_phases = np.exp(1j * phases)
        order_param = np.abs(np.mean(complex_phases))
        return order_param
    
    def _compute_phase_coherence(self, phases):
        """Compute phase coherence (alternative synchronization measure)."""
        # Normalize phases to [-π, π] for variance calculation
        phases_norm = (phases + np.pi) % (2 * np.pi) - np.pi
        # Compute variance of phases
        phase_variance = np.var(phases_norm)
        # Coherence is inversely related to variance
        coherence = np.exp(-phase_variance)
        return coherence
    
    def _kuramoto_dynamics(self, phases, natural_frequencies, coupling_matrix, control_inputs):
        """Compute the Kuramoto dynamics."""
        # Compute coupling term: Σⱼ K_ij sin(θⱼ - θᵢ)
        coupling_term = np.zeros_like(phases)
        for i in range(self.n_oscillators):
            for j in range(self.n_oscillators):
                if i != j and coupling_matrix[i, j] > 0:
                    coupling_term[i] += coupling_matrix[i, j] * np.sin(phases[j] - phases[i])
        
        # Total dynamics: dθᵢ/dt = ωᵢ + control_inputᵢ + coupling_termᵢ
        dphases_dt = natural_frequencies + control_inputs + coupling_term
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
        
        # Initialize phases randomly in [-π, π]
        self.phases = (self.np_random.random(self.n_oscillators) * 2 - 1) * np.pi
        
        # Initialize natural frequencies
        self.natural_frequencies = self.np_random.random(self.n_oscillators) * \
                                  (self.natural_freq_range[1] - self.natural_freq_range[0]) + \
                                  self.natural_freq_range[0]
        
        # Initialize coupling strengths (only in dynamic mode)
        if self.coupling_mode == "dynamic":
            self.coupling_strengths = self.np_random.random(self.n_couplings) * \
                                     (self.coupling_range[1] - self.coupling_range[0]) + \
                                     self.coupling_range[0]
        else:
            self.coupling_strengths = None

        # Initialize control inputs to zero
        self.control_inputs = np.zeros(self.n_oscillators)
        
        self.step_count = 0
        self.phase_history = [self.phases.copy()]
        
        # Create observation based on coupling mode
        if self.coupling_mode == "dynamic":
            assert self.coupling_strengths is not None
            observation = np.concatenate([
                self.phases,
                self.natural_frequencies,
                self.coupling_strengths,
                self.control_inputs
            ])
        else:  # constant mode
            observation = np.concatenate([
                self.phases,
                self.natural_frequencies,
                self.control_inputs
            ])
        
        assert self.phases is not None
        info = {
            'order_parameter': self._compute_order_parameter(self.phases),
            'phase_coherence': self._compute_phase_coherence(self.phases),
            'natural_frequencies': self.natural_frequencies.copy(),
            'phases': self.phases.copy(),
            'device': self.device,
            'n_agents': self.n_agents
        }
        
        return observation.astype(np.float32), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        assert self.phases is not None and self.natural_frequencies is not None and self.control_inputs is not None
        
        if self.coupling_mode == "dynamic":
            # Split action into control inputs and coupling strengths
            control_actions = action[:self.n_oscillators]
            coupling_actions = action[self.n_oscillators:]

            # Update state with validation
            self.control_inputs = np.clip(control_actions, self.control_input_range[0], self.control_input_range[1])
            self.coupling_strengths = np.clip(coupling_actions, self.coupling_range[0], self.coupling_range[1])
            
            # Convert actions to coupling matrix
            coupling_matrix = self._coupling_matrix_from_actions(self.coupling_strengths)
        else:  # constant mode
            # Only control inputs in action
            control_actions = action
            
            # Update state
            self.control_inputs = np.clip(control_actions, self.control_input_range[0], self.control_input_range[1])
            
            # Use constant coupling matrix
            coupling_matrix = self.coupling_matrix
        
        # Integrate Kuramoto dynamics
        dphases_dt = None
        if self.integration_method == "euler":
            dphases_dt = self._kuramoto_dynamics(self.phases, self.natural_frequencies, coupling_matrix, self.control_inputs)
            self.phases += dphases_dt * self.dt
        elif self.integration_method == "rk4":
            self.phases, dphases_dt = self._integrate_rk4(self.phases, self.natural_frequencies, coupling_matrix, self.control_inputs)
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")

        # Add noise if specified
        if self.noise_std > 0:
            noise = self.np_random.normal(0, self.noise_std, self.n_oscillators)
            assert self.phases is not None
            self.phases += noise
        
        # Safe phase wrapping
        assert self.phases is not None
        self.phases = (self.phases + np.pi) % (2 * np.pi) - np.pi
        
        # Store phase history for rendering
        self.phase_history.append(self.phases.copy())
        
        # Compute synchronization measures
        order_parameter = self._compute_order_parameter(self.phases)
        phase_coherence = self._compute_phase_coherence(self.phases)
        
        # Create observation based on coupling mode
        if self.coupling_mode == "dynamic":
            assert self.coupling_strengths is not None
            observation = np.concatenate([
                self.phases,
                self.natural_frequencies,
                self.coupling_strengths,
                self.control_inputs
            ])
        else:  # constant mode
            observation = np.concatenate([
                self.phases,
                self.natural_frequencies,
                self.control_inputs
            ])
        
        # Compute reward
        reward = 0.0
        if self.reward_type == "frequency_synchronization":
            agent_rewards = -np.abs(dphases_dt - self.target_frequency)
            reward = np.mean(agent_rewards)
        elif self.reward_type == "order_parameter":
            reward = order_parameter
        elif self.reward_type == "phase_coherence":
            reward = phase_coherence
        elif self.reward_type == "combined":
            reward = order_parameter + phase_coherence
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
        # Check if episode is done
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # Additional termination condition: perfect synchronization
        if order_parameter > 0.99:
            if self.reward_type != "frequency_synchronization":
                reward += 10.0  # Bonus for achieving synchronization
            done = True
        
        info = {
            'order_parameter': order_parameter,
            'phase_coherence': phase_coherence,
            'step_count': self.step_count,
            'natural_frequencies': self.natural_frequencies.copy(),
            'phases': self.phases.copy(),
            'coupling_matrix': coupling_matrix.copy() if coupling_matrix is not None else None,
            'device': self.device,
            'n_agents': self.n_agents
        }
        
        if dphases_dt is not None:
            info['dphases_dt'] = dphases_dt.copy()
        
        return observation.astype(np.float32), reward, done, False, info
    
    def render(self):
        """Render the current state of the oscillators."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        """Render a single frame showing oscillator phases."""
        assert self.phases is not None and self.natural_frequencies is not None

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.ax.set_xlim(-1.5, 1.5)
            self.ax.set_ylim(-1.5, 1.5)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_title('Kuramoto Oscillators')
            
            # Draw unit circle
            circle = Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
            self.ax.add_patch(circle)
        
        assert self.ax is not None
        self.ax.clear()
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw unit circle
        circle = Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
        self.ax.add_patch(circle)
        
        # Plot oscillators on unit circle
        x_positions = np.cos(self.phases)
        y_positions = np.sin(self.phases)
        
        # Color oscillators based on their natural frequency
        colors = plt.get_cmap('viridis')((self.natural_frequencies - self.natural_freq_range[0]) /
                               (self.natural_freq_range[1] - self.natural_freq_range[0]))
        
        scatter = self.ax.scatter(x_positions, y_positions, c=colors, s=100, alpha=0.7)
        
        # Add oscillator labels
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            self.ax.annotate(f'O{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Show order parameter
        order_param = self._compute_order_parameter(self.phases)
        self.ax.set_title(f'Kuramoto Oscillators - Order Parameter: {order_param:.3f}')
        
        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img
    
    def close(self):
        """Close the environment and cleanup."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None 