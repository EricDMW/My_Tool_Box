from gymnasium.envs.registration import register as gym_register
from .kuramoto_env import KuramotoOscillatorEnv
from .kuramoto_env_torch import KuramotoOscillatorEnvTorch

# Register the original NumPy-based environment
gym_register(
    id='KuramotoOscillator-v0',
    entry_point='env_lib.kos_env.kuramoto_env:KuramotoOscillatorEnv',
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={}
)

# Register a simpler version with fewer oscillators
gym_register(
    id='KuramotoOscillator-v1',
    entry_point='env_lib.kos_env.kuramoto_env:KuramotoOscillatorEnv',
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'n_oscillators': 5}
)

# Register constant coupling matrix version (NumPy-based)
gym_register(
    id='KuramotoOscillator-Constant-v0',
    entry_point='env_lib.kos_env.kuramoto_env:KuramotoOscillatorEnv',
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'n_oscillators': 6, 'coupling_mode': 'constant', 'constant_coupling_matrix': None}
)

# Register frequency synchronization version with constant coupling (NumPy-based)
gym_register(
    id='KuramotoOscillator-FreqSync-Constant-v0',
    entry_point='env_lib.kos_env.kuramoto_env:KuramotoOscillatorEnv',
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'n_oscillators': 6, 'coupling_mode': 'constant', 'reward_type': 'frequency_synchronization', 'target_frequency': 2.0}
)

# Register the PyTorch-based environment
gym_register(
    id='KuramotoOscillatorTorch-v0',
    entry_point='env_lib.kos_env.kuramoto_env_torch:KuramotoOscillatorEnvTorch',
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={}
)

# Register PyTorch version with multi-agent support
gym_register(
    id='KuramotoOscillatorTorch-v1',
    entry_point='env_lib.kos_env.kuramoto_env_torch:KuramotoOscillatorEnvTorch',
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'n_oscillators': 8, 'n_agents': 4, 'device': 'cpu'}
)

# Register GPU version
gym_register(
    id='KuramotoOscillatorTorch-v2',
    entry_point='env_lib.kos_env.kuramoto_env_torch:KuramotoOscillatorEnvTorch',
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'n_oscillators': 10, 'n_agents': 1, 'device': 'cuda', 'integration_method': 'rk4'}
)

# Register constant coupling matrix version (PyTorch-based)
gym_register(
    id='KuramotoOscillatorTorch-Constant-v0',
    entry_point='env_lib.kos_env.kuramoto_env_torch:KuramotoOscillatorEnvTorch',
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'n_oscillators': 6, 'coupling_mode': 'constant', 'constant_coupling_matrix': None}
)

# Register frequency synchronization version with constant coupling (PyTorch-based)
gym_register(
    id='KuramotoOscillatorTorch-FreqSync-Constant-v0',
    entry_point='env_lib.kos_env.kuramoto_env_torch:KuramotoOscillatorEnvTorch',
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'n_oscillators': 6, 'coupling_mode': 'constant', 'reward_type': 'frequency_synchronization', 'target_frequency': 2.0}
)

__all__ = ['KuramotoOscillatorEnv', 'KuramotoOscillatorEnvTorch']

def register():
    """Entry point for gymnasium environment registration.
    This function is called automatically when the package is installed.
    """
    # All environments are already registered above when the module is imported
    pass
