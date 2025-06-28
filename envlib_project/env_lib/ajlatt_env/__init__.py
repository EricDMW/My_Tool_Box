from .env import make as ajlatt_env
from .plot_map import load_map

from gymnasium.envs.registration import register as gym_register

# Register the AJLATT environment
gym_register(
    id='AJLATT-v0',
    entry_point='env_lib.ajlatt_env.env.ajlatt_agent:make',
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={}
)

# Register a version with custom parameters
gym_register(
    id='AJLATT-v1',
    entry_point='env_lib.ajlatt_env.env.ajlatt_agent:make',
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'figID': 0}
)

__version__ = "0.1.0"

__all__ = [
    "ajlatt_env",
    "load_map",
]
