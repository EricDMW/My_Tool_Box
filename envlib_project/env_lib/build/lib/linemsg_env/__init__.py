from gymnasium.envs.registration import register

register(
    id="LineMsg-v0",
    entry_point="env_lib.linemsg_env.linemsg_env:LineMsgEnv",
    max_episode_steps=50,
)

from .linemsg_env import LineMsgEnv 