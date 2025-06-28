from gymnasium.envs.registration import register

register(
    id="Pistonball-v0",
    entry_point="env_lib.pistonball_env.pistonball_env:PistonballEnv",
    max_episode_steps=125,
)

from .pistonball_env import PistonballEnv 