"""Keyboard control for :class:`~env_lib.pistonball_env.PistonballEnv`.

Create the environment with ``render_mode="human"`` so that a pygame window
exists and receives the key presses, then call the policy once per step::

    env = PistonballEnv(n_pistons=8, render_mode="human")
    policy = ManualPolicy(env)
    obs, info = env.reset()
    while policy.running:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

``examples/pistonball_demo.py --manual`` runs this loop.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

__all__ = ["ManualPolicy"]

_logger = logging.getLogger(__name__)


class ManualPolicy:
    """Control one piston at a time with the keyboard.

    Controls: ``W`` / ``Up`` raise the selected piston, ``S`` / ``Down`` lower
    it (hold the key to keep moving), ``A`` / ``Left`` and ``D`` / ``Right``
    select the previous / next piston, ``Backspace`` resets the environment,
    ``Esc`` or closing the window sets :attr:`running` to ``False``.

    Parameters
    ----------
    env:
        A :class:`~env_lib.pistonball_env.PistonballEnv` (or a wrapper of one)
        created with ``render_mode="human"``.
    agent_id:
        Index of the initially selected piston.
    show_obs:
        Log the selected piston's observation row at DEBUG level.

    Attributes
    ----------
    selected_piston:
        Index of the piston that the keys move.
    running:
        ``False`` once the user pressed ``Esc`` or closed the window.
    """

    def __init__(self, env: Any, agent_id: int = 0, show_obs: bool = False):
        import pygame

        self._pg = pygame
        self.env = env
        base = getattr(env, "unwrapped", env)
        self.n_pistons = int(base.n_pistons)
        if not 0 <= agent_id < self.n_pistons:
            raise ValueError(f"agent_id must be in [0, {self.n_pistons - 1}], got {agent_id}")
        self.agent_id = agent_id
        self.show_obs = show_obs
        self.continuous = bool(base.continuous)
        self.selected_piston = agent_id
        self.running = True
        # Neutral action: 0.0 (continuous) or 1 = stay (discrete).
        if self.continuous:
            self.default_action = np.zeros(self.n_pistons, dtype=np.float32)
        else:
            self.default_action = np.ones(self.n_pistons, dtype=np.int64)
        self.action_mapping = {
            pygame.K_w: 1.0,
            pygame.K_UP: 1.0,
            pygame.K_s: -1.0,
            pygame.K_DOWN: -1.0,
        }

    def __call__(self, observation: np.ndarray | None = None, agent: Any = None) -> np.ndarray:
        """Read the keyboard and return a joint action for all pistons.

        Parameters
        ----------
        observation:
            Current observation (only used when ``show_obs`` is set).
        agent:
            Ignored; kept for API compatibility.

        Returns
        -------
        numpy.ndarray
            Action of shape ``(n_pistons,)`` in the environment's action format.
        """
        pg = self._pg
        action = self.default_action.copy()
        if not pg.display.get_init() or pg.display.get_surface() is None:
            return action

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.running = False
                elif event.key == pg.K_BACKSPACE:
                    self.env.reset()
                elif event.key in (pg.K_a, pg.K_LEFT):
                    self._select(self.selected_piston - 1)
                elif event.key in (pg.K_d, pg.K_RIGHT):
                    self._select(self.selected_piston + 1)

        pressed = pg.key.get_pressed()
        direction = sum(v for key, v in self.action_mapping.items() if pressed[key])
        direction = float(np.clip(direction, -1.0, 1.0))
        if self.continuous:
            action[self.selected_piston] = direction
        else:
            action[self.selected_piston] = int(direction) + 1
        if self.show_obs and observation is not None:
            _logger.debug(
                "piston %d observation: %s", self.selected_piston, observation[self.selected_piston]
            )
        return action

    def _select(self, index: int) -> None:
        self.selected_piston = index % self.n_pistons
        _logger.info("Selected piston: %d", self.selected_piston)

    @property
    def available_agents(self) -> list[str]:
        """Agent names of the controlled environment."""
        return list(getattr(self.env, "unwrapped", self.env).agents)
