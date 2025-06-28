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
                 dt: float = 0.01,
                 max_steps: int = 1000,
                 coupling_range: Tuple[float, float] = (0.0, 5.0),
                 control_input_range: Tuple[float, float] = (-1.0, 1.0),
                 natural_freq_range: Tuple[float, float] = (0.5, 2.0),
                 noise_std: float = 0.0,
                 reward_type: str = "order_parameter",
                 target_frequency: float = 1.0,
                 adj_matrix: Optional[np.ndarray] = None,
                 render_mode: Optional[str] = None,
                 coupling_mode: str = "dynamic",
                 constant_coupling_matrix: Optional[np.ndarray] = None):
        """
        Initialize the Kuramoto oscillator environment.
        
        Args:
            n_oscillators: Number of oscillators in the system
            dt: Time step for integration
            max_steps: Maximum number of steps per episode
            coupling_range: Range for coupling strength values (used only in dynamic mode)
            control_input_range: Range for external control input values
            natural_freq_range: Range for natural frequencies
            noise_std: Standard deviation of noise added to dynamics
            reward_type: Reward type ('order_parameter', 'phase_coherence', 'combined', 'frequency_synchronization')
            target_frequency: The target frequency for the 'frequency_synchronization' reward.
            adj_matrix: An adjacency matrix to define a custom topology.
            render_mode: Rendering mode ('human' or 'rgb_array')
            coupling_mode: Either 'dynamic' (coupling strengths controlled by actions) or 'constant' (fixed coupling matrix)
            constant_coupling_matrix: Fixed coupling matrix to use when coupling_mode='constant'
        """
        super().__init__()
        
        self.n_oscillators = n_oscillators
        self.dt = dt
        self.max_steps = max_steps
        self.coupling_range = coupling_range
        self.control_input_range = control_input_range
        self.natural_freq_range = natural_freq_range
        self.noise_std = noise_std
        self.reward_type = reward_type
        self.target_frequency = target_frequency
        self.adj_matrix = adj_matrix
        self.render_mode = render_mode
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
        else: # Default to fully connected if no matrix is provided
            self.topology_matrix = np.ones((self.n_oscillators, self.n_oscillators))
            np.fill_diagonal(self.topology_matrix, 0)
    
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
        """Compute the Kuramoto dynamics for all oscillators."""
        # Natural frequency term and control input
        dphases_dt = natural_frequencies + control_inputs
        
        # Vectorized computation of coupling term
        phase_diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
        sin_phase_diffs = np.sin(phase_diffs)
        coupling_effects = coupling_matrix * sin_phase_diffs
        coupling_sum = np.sum(coupling_effects, axis=1)
        
        dphases_dt += coupling_sum
            
        return dphases_dt
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize phases randomly in [-π, π]
        self.phases = self.np_random.uniform(-np.pi, np.pi, self.n_oscillators)
        
        # Initialize natural frequencies
        self.natural_frequencies = self.np_random.uniform(
            self.natural_freq_range[0], 
            self.natural_freq_range[1], 
            self.n_oscillators
        )
        
        # Initialize coupling strengths (only in dynamic mode)
        if self.coupling_mode == "dynamic":
            self.coupling_strengths = self.np_random.uniform(
                self.coupling_range[0],
                self.coupling_range[1],
                self.n_couplings
            )
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
            'phases': self.phases.copy()
        }
        
        return observation.astype(np.float32), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        assert self.phases is not None and self.natural_frequencies is not None and self.control_inputs is not None
        
        if self.coupling_mode == "dynamic":
            # Split action into control inputs and coupling strengths
            control_actions = action[:self.n_oscillators]
            coupling_actions = action[self.n_oscillators:]

            # Update state
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
        
        # Integrate Kuramoto dynamics using Euler method
        dphases_dt = self._kuramoto_dynamics(self.phases, self.natural_frequencies, coupling_matrix, self.control_inputs)
        assert self.phases is not None
        self.phases += dphases_dt * self.dt

        # Add noise if specified
        if self.noise_std > 0:
            noise = self.np_random.normal(0, self.noise_std, self.n_oscillators)
            assert self.phases is not None
            self.phases += noise
        
        # Normalize phases to [-π, π]
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
            'dphases_dt': dphases_dt.copy()
        }
        
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