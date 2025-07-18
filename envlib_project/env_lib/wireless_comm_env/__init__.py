from gymnasium.envs.registration import register as gym_register
from .wireless_comm_env import WirelessCommEnv
from .example_render_progress import run_and_render_animation

# Register the wireless communication environment
gym_register(
    id='WirelessComm-v0',
    entry_point='env_lib.wireless_comm_env:WirelessCommEnv',
    max_episode_steps=50,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={}
)

# Register a smaller version
gym_register(
    id='WirelessComm-v1',
    entry_point='env_lib.wireless_comm_env:WirelessCommEnv',
    max_episode_steps=50,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'grid_x': 4, 'grid_y': 4}
)

__all__ = ['WirelessCommEnv']

def register():
    """Entry point for gymnasium environment registration.
    This function is called automatically when the package is installed.
    """
    # All environments are already registered above when the module is imported
    pass 