"""Pistonball: a cooperative multi-agent physics environment.

A row of ``n_pistons`` pistons sits at the bottom of a walled arena and a ball
is dropped near the right wall. Every piston is an agent that moves up or down;
together they have to roll the ball to the left wall (the goal). The physics is
simulated with `pymunk <https://www.pymunk.org>`_; ``pygame`` is only needed for
rendering and is imported lazily by :mod:`env_lib.pistonball_env.rendering`.

The environment exposes the whole team through a single Gymnasium interface:
actions are stacked per piston, observations have shape ``(n_pistons, 7)`` and
the scalar reward is the sum of the per-piston (local) rewards, which are also
returned in ``info["local_rewards"]`` / ``info["agent_rewards"]``.

Observability is local: a piston sees the ball only when it is within
``kappa`` hops of the piston under the ball.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.utils import EzPickle, seeding

from env_lib.errors import ResetNeededError
from env_lib.utils.rendering import validate_render_mode

try:
    import pymunk
except ImportError as exc:  # pragma: no cover - depends on the installation
    raise ImportError(
        "PistonballEnv requires pymunk (and pygame for rendering): "
        'pip install "my-tool-box[pistonball]"'
    ) from exc

__all__ = ["FPS", "PistonballEnv", "safe_physics_step", "validate_ball_position"]

FPS = 20
"""Simulation and render frame rate (one physics step of ``1 / FPS`` s per env step)."""

_MAX_BALL_SPEED = 1000.0  # px/s; caps the ball speed before every physics step


def validate_ball_position(
    x: float, y: float, screen_width: float, screen_height: float, ball_radius: float
) -> tuple[float, float]:
    """Clamp a ball centre so that the ball stays inside the screen.

    Parameters
    ----------
    x, y:
        Ball centre in pixels.
    screen_width, screen_height:
        Screen size in pixels.
    ball_radius:
        Ball radius in pixels.

    Returns
    -------
    tuple of float
        The clamped ``(x, y)``.
    """
    x = min(max(float(x), ball_radius), screen_width - ball_radius)
    y = min(max(float(y), ball_radius), screen_height - ball_radius)
    return x, y


def safe_physics_step(space: Any, dt: float, max_velocity: float = _MAX_BALL_SPEED) -> int:
    """Cap the speed of every body in ``space`` and advance it by ``dt``.

    Parameters
    ----------
    space:
        A :class:`pymunk.Space`.
    dt:
        Time step in seconds.
    max_velocity:
        Speed limit in px/s applied to every body before stepping.

    Returns
    -------
    int
        Always ``1`` (the number of physics steps taken). Errors raised by
        pymunk propagate to the caller.
    """
    for body in list(space.bodies):
        velocity = body.velocity
        if velocity.length > max_velocity:
            body.velocity = velocity.normalized() * max_velocity
    space.step(dt)
    return 1


def _as_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _as_float(name: str, value: Any, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be {relation} {minimum}, got {value}")
    return value


class PistonballEnv(Env, EzPickle):
    """Cooperative Pistonball with ``n_pistons`` piston agents.

    Parameters
    ----------
    n_pistons:
        Number of piston agents (>= 1). The screen is ``160 + 40 * n_pistons``
        pixels wide and 560 pixels high.
    time_penalty:
        Reward added to every piston at every step.
    continuous:
        ``True`` for continuous actions in ``[-1, 1]`` per piston, ``False``
        for discrete actions ``{0: down, 1: stay, 2: up}`` per piston.
    random_drop:
        Randomise the initial ball position (+-30 px horizontally, +-15 px vertically).
    random_rotate:
        Give the ball a random initial angular velocity in ``[-6 pi, 6 pi]`` rad/s.
    ball_mass, ball_friction, ball_elasticity:
        Physical properties of the ball.
    max_cycles:
        Episode length; ``truncated`` becomes ``True`` after this many steps.
    render_mode:
        ``None``, ``"human"`` (pygame window) or ``"rgb_array"``.
    movement_penalty:
        Reward per piston per ``pixels_per_position`` (4 px) of movement,
        typically negative; ``0`` disables it.
    movement_penalty_threshold:
        Movements of at most this many pixels are not penalised.
    kappa:
        Observation radius in hops: a piston observes the ball when
        ``|i - ball_piston_index| <= kappa``.
    terminated_condition:
        End the episode (``terminated=True``) when the ball reaches the left wall.
    leftmost_piston_reward:
        Extra reward for piston 0 when the ball reaches the left wall.
    termination_reward:
        Extra reward for every piston that observes the ball when it reaches
        the left wall.

    Notes
    -----
    Coordinates are pygame screen pixels (``y`` grows downwards). With
    ``wall_width = 80``, ``piston_width = 40`` and ``piston_radius = 5``, the
    ball piston index is ``clip(int((x - 85) / 40), 0, n_pistons - 1)``.

    Observation (``float32``, shape ``(n_pistons, 7)``), row ``i``:

    ====  ========================================================================
    0     piston height ``(y_i - mid_piston_y) / 32`` in ``[-1, 1]`` (``+1`` = lowest)
    1     piston x position ``(5 + 40 i) / (40 n_pistons)`` in ``[0, 1]``
    2     ball x ``(x - 80) / (screen_width - 160)``           (0 if not observed)
    3     ball y ``(y - 80) / (screen_height - 160)``          (0 if not observed)
    4     ball x velocity / 15                                 (0 if not observed)
    5     ball y velocity / 8                                  (0 if not observed)
    6     ball angular velocity / 8                            (0 if not observed)
    ====  ========================================================================

    Reward: with ``b_t = int(ball_x - ball_radius)`` (the ball's left edge) the
    ball term is ``0.5 (b_{t-1} - b_t)`` if the ball moved left and
    ``b_{t-1} - b_t`` otherwise. Piston ``i`` receives the ball term only if it
    observes the ball's left edge at ``t - 1`` or at ``t`` (kappa rule applied
    to ``b``), plus ``time_penalty``, plus ``termination_reward`` (observers of
    the ball centre) and ``leftmost_piston_reward`` (piston 0) on reaching the
    goal, plus the movement penalty. The returned reward is the sum over pistons.

    Examples
    --------
    >>> env = PistonballEnv(n_pistons=10, kappa=2)
    >>> obs, info = env.reset(seed=0)
    >>> obs.shape
    (10, 7)
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    >>> info["agent_rewards"].shape
    (10,)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "Pistonball-v0",
        "render_fps": FPS,
    }

    def __init__(
        self,
        n_pistons: int = 20,
        time_penalty: float = -0.1,
        continuous: bool = True,
        random_drop: bool = True,
        random_rotate: bool = True,
        ball_mass: float = 0.75,
        ball_friction: float = 0.3,
        ball_elasticity: float = 1.5,
        max_cycles: int = 125,
        render_mode: str | None = None,
        movement_penalty: float = 0.0,
        movement_penalty_threshold: float = 0.01,
        kappa: int = 1,
        terminated_condition: bool = True,
        leftmost_piston_reward: float = 0.0,
        termination_reward: float = 0.5,
    ):
        EzPickle.__init__(
            self,
            n_pistons=n_pistons,
            time_penalty=time_penalty,
            continuous=continuous,
            random_drop=random_drop,
            random_rotate=random_rotate,
            ball_mass=ball_mass,
            ball_friction=ball_friction,
            ball_elasticity=ball_elasticity,
            max_cycles=max_cycles,
            render_mode=render_mode,
            movement_penalty=movement_penalty,
            movement_penalty_threshold=movement_penalty_threshold,
            kappa=kappa,
            terminated_condition=terminated_condition,
            leftmost_piston_reward=leftmost_piston_reward,
            termination_reward=termination_reward,
        )
        validate_render_mode(render_mode, self.metadata["render_modes"])

        # Configuration.
        self.n_pistons = _as_int("n_pistons", n_pistons, 1)
        self.time_penalty = _as_float("time_penalty", time_penalty)
        self.continuous = bool(continuous)
        self.random_drop = bool(random_drop)
        self.random_rotate = bool(random_rotate)
        self.ball_mass = _as_float("ball_mass", ball_mass, 0.0, strict=True)
        self.ball_friction = _as_float("ball_friction", ball_friction, 0.0)
        self.ball_elasticity = _as_float("ball_elasticity", ball_elasticity, 0.0)
        self.max_cycles = _as_int("max_cycles", max_cycles, 1)
        self.render_mode = render_mode
        self.movement_penalty = _as_float("movement_penalty", movement_penalty)
        self.movement_penalty_threshold = _as_float(
            "movement_penalty_threshold", movement_penalty_threshold
        )
        self.kappa = _as_int("kappa", kappa, 0)
        self.terminated_condition = bool(terminated_condition)
        self.leftmost_piston_reward = _as_float("leftmost_piston_reward", leftmost_piston_reward)
        self.termination_reward = _as_float("termination_reward", termination_reward)

        # Geometry (pixels) and physics constants.
        self.dt = 1.0 / FPS
        self.piston_head_height = 11
        self.piston_width = 40
        self.piston_height = 40
        self.piston_body_height = 23
        self.piston_radius = 5
        self.wall_width = 40 * 2
        self.ball_radius = 40
        self.screen_width = (2 * self.wall_width) + (self.piston_width * self.n_pistons)
        self.screen_height = 560
        self.maximum_piston_y = (
            self.screen_height - self.wall_width - (self.piston_height - self.piston_head_height)
        )
        self.pixels_per_position = 4
        self.n_piston_positions = 16
        self.piston_y_half_range = 0.5 * self.pixels_per_position * self.n_piston_positions
        self.mid_piston_y = self.maximum_piston_y - self.piston_y_half_range
        self.minimum_piston_y = (
            self.maximum_piston_y - self.n_piston_positions * self.pixels_per_position
        )

        # Precomputed per-piston constants.
        self._piston_index = np.arange(self.n_pistons)
        self._piston_x = (
            self.wall_width + self.piston_radius + self.piston_width * self._piston_index
        ).astype(np.float64)
        arena_width = self.screen_width - 2 * self.wall_width
        arena_height = self.screen_height - 2 * self.wall_width
        self._obs_template = np.zeros((self.n_pistons, 7), dtype=np.float32)
        self._obs_template[:, 1] = (self._piston_x - self.wall_width) / arena_width
        self._arena_size = (float(arena_width), float(arena_height))
        self._initial_displacements = np.arange(
            0, 0.5 * self.pixels_per_position * self.n_piston_positions, self.pixels_per_position
        )

        # Agents (kept for PettingZoo-style bookkeeping).
        self.agents = [f"piston_{i}" for i in range(self.n_pistons)]
        self.agent_name_mapping = dict(zip(self.agents, range(self.n_pistons)))

        # Spaces.
        if self.continuous:
            self.action_space = Box(low=-1.0, high=1.0, shape=(self.n_pistons,), dtype=np.float32)
        else:
            self.action_space = MultiDiscrete(np.full(self.n_pistons, 3, dtype=np.int64))
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.n_pistons, 7), dtype=np.float32
        )
        self.state_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )

        # Simulation state (created by reset()).
        self.space: pymunk.Space | None = None
        self.ball: pymunk.Body | None = None
        self.pistonList: list[pymunk.Body] = []
        self.piston_pos_y = np.full(self.n_pistons, self.mid_piston_y, dtype=np.float64)
        self.last_ball_x: int | None = None
        self.lastX: int | None = None
        self.frames = 0
        self.terminate = False
        self.truncate = False
        self._episode_return = 0.0

        # Rendering state: nothing is created before the first render() call.
        self._renderer: Any = None
        self._warned_no_render_mode = False

    # ------------------------------------------------------------------
    # Seeding (deprecated API)
    # ------------------------------------------------------------------
    def seed(self, seed: int | None = None) -> list[int]:
        """Reseed the environment's random generator.

        .. deprecated::
            Use ``reset(seed=...)`` instead.
        """
        warnings.warn(
            "PistonballEnv.seed() is deprecated; use reset(seed=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._np_random, seed = seeding.np_random(seed)
        self._np_random_seed = seed
        return [seed]

    # ------------------------------------------------------------------
    # Physics construction
    # ------------------------------------------------------------------
    def add_walls(self) -> None:
        """Add the four arena walls to :attr:`space`."""
        if self.space is None:
            return
        left, right = self.wall_width, self.screen_width - self.wall_width
        top, bottom = self.wall_width, self.screen_height - self.wall_width
        walls = [
            pymunk.Segment(self.space.static_body, (left, top), (right, top), 1),
            pymunk.Segment(self.space.static_body, (left, top), (left, bottom), 1),
            pymunk.Segment(self.space.static_body, (left, bottom), (right, bottom), 1),
            pymunk.Segment(self.space.static_body, (right, top), (right, bottom), 1),
        ]
        for wall in walls:
            wall.friction = 0.64
            self.space.add(wall)

    def add_ball(self, x: float, y: float) -> pymunk.Body:
        """Add the ball at ``(x, y)`` (clamped to the screen) and return its body."""
        x, y = validate_ball_position(x, y, self.screen_width, self.screen_height, self.ball_radius)
        inertia = pymunk.moment_for_circle(self.ball_mass, 0, self.ball_radius, (0, 0))
        body = pymunk.Body(self.ball_mass, inertia)
        body.position = x, y
        if self.random_rotate:
            body.angular_velocity = self.np_random.uniform(-6 * math.pi, 6 * math.pi)
        shape = pymunk.Circle(body, self.ball_radius, (0, 0))
        shape.friction = self.ball_friction
        shape.elasticity = self.ball_elasticity
        self.space.add(body, shape)
        return body

    def add_piston(self, space: pymunk.Space, x: float, y: float) -> pymunk.Body:
        """Add a kinematic piston whose head segment starts at ``(x, y)``."""
        piston = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        piston.position = x, y
        segment = pymunk.Segment(
            piston, (0, 0), (self.piston_width - (2 * self.piston_radius), 0), self.piston_radius
        )
        segment.friction = 0.64
        space.add(piston, segment)
        return piston

    def move_piston(self, piston: pymunk.Body, v: float) -> None:
        """Move one piston by ``v`` positions (positive = up), clamped to its travel."""
        y = piston.position[1] - v * self.pixels_per_position
        y = min(max(y, self.minimum_piston_y), self.maximum_piston_y)
        piston.position = (piston.position[0], y)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Parameters
        ----------
        seed:
            Seed for the environment's random generator.
        options:
            Unused; accepted for Gymnasium compatibility.

        Returns
        -------
        observation : numpy.ndarray
            ``float32`` array of shape ``(n_pistons, 7)``.
        info : dict
            Empty dictionary.
        """
        super().reset(seed=seed)
        rng = self.np_random

        self.space = pymunk.Space()
        self.add_walls()
        self.space.gravity = (0.0, 750.0)
        self.space.collision_bias = 0.0001
        self.space.iterations = 10

        # One draw per piston, in piston order (identical stream to the original loop).
        heights = self.maximum_piston_y - rng.choice(
            self._initial_displacements, size=self.n_pistons
        )
        self.pistonList = []
        for x, y in zip(self._piston_x.tolist(), heights.tolist()):
            piston = self.add_piston(self.space, x, y)
            piston.velocity = (0, 0)
            self.pistonList.append(piston)
        self.piston_pos_y = np.asarray(heights, dtype=np.float64)

        horizontal_offset_range = 30
        vertical_offset_range = 15
        horizontal_offset = vertical_offset = 0
        if self.random_drop:
            vertical_offset = int(rng.integers(-vertical_offset_range, vertical_offset_range + 1))
            horizontal_offset = int(
                rng.integers(-horizontal_offset_range, horizontal_offset_range + 1)
            )
        ball_x = (
            self.screen_width
            - self.wall_width
            - self.ball_radius
            - horizontal_offset_range
            + horizontal_offset
        )
        ball_y = (
            self.screen_height
            - self.wall_width
            - self.piston_body_height
            - self.ball_radius
            - (0.5 * self.pixels_per_position * self.n_piston_positions)
            - vertical_offset_range
            + vertical_offset
        )
        ball_x = max(ball_x, self.wall_width + self.ball_radius + 1)
        self.ball = self.add_ball(ball_x, ball_y)
        self.ball.angle = 0
        self.ball.velocity = (0, 0)
        if self.random_rotate:
            # add_ball() already drew one angular velocity; this second draw is the one
            # that is used. Both are kept so that seeded episodes match earlier versions.
            self.ball.angular_velocity = rng.uniform(-6 * math.pi, 6 * math.pi)

        self.last_ball_x = int(self.ball.position[0] - self.ball_radius)
        self.lastX = self.last_ball_x
        self.frames = 0
        self.terminate = False
        self.truncate = False
        self._episode_return = 0.0

        if self._renderer is not None:
            self._renderer.reset()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one joint action.

        Parameters
        ----------
        action:
            Array of shape ``(n_pistons,)``: values in ``[-1, 1]`` (continuous,
            clipped) or in ``{0, 1, 2}`` (discrete: down, stay, up).

        Returns
        -------
        observation, reward, terminated, truncated, info
            ``reward`` is the team reward (sum of local rewards). ``info``
            holds ``"local_rewards"`` and ``"agent_rewards"`` (per-piston
            rewards, ``float64`` arrays of shape ``(n_pistons,)``) and
            ``"total_reward"``.

        Raises
        ------
        ValueError
            If the action has the wrong shape or invalid values.
        RuntimeError
            If :meth:`reset` has not been called.
        """
        if self.space is None or self.ball is None:
            raise ResetNeededError("PistonballEnv.step() called before reset()")
        displacement = self._action_to_displacement(action)

        # Move the pistons (kinematic bodies are teleported to their new height).
        prev_y = self.piston_pos_y
        new_y = np.clip(
            prev_y - displacement * self.pixels_per_position,
            self.minimum_piston_y,
            self.maximum_piston_y,
        )
        for i in np.flatnonzero(new_y != prev_y).tolist():
            self.pistonList[i].position = (self._piston_x[i], new_y[i])
        self.piston_pos_y = new_y

        # Advance the physics with the ball speed capped.
        ball = self.ball
        velocity = ball.velocity
        if velocity.length > _MAX_BALL_SPEED:
            ball.velocity = velocity.normalized() * _MAX_BALL_SPEED
        self.space.step(self.dt)
        x, y = ball.position
        clamped = validate_ball_position(
            x, y, self.screen_width, self.screen_height, self.ball_radius
        )
        if clamped != (x, y):
            ball.position = clamped
            x, y = clamped

        # Termination: the ball's left edge reaches the left wall within one step.
        ball_min_x = int(x - self.ball_radius)
        ball_next_x = x - self.ball_radius + ball.velocity[0] * self.dt
        reached_goal = self.terminated_condition and ball_next_x <= self.wall_width + 0.5
        if reached_goal:
            self.terminate = True

        # Local rewards.
        local_rewards = self._local_rewards(self.last_ball_x, ball_min_x)
        if reached_goal:
            local_rewards[self._observer_slice(x)] += self.termination_reward
            local_rewards[0] += self.leftmost_piston_reward
        if self.movement_penalty != 0.0:
            moved = np.abs(new_y - prev_y)
            local_rewards += np.where(
                moved > self.movement_penalty_threshold,
                self.movement_penalty * (moved / self.pixels_per_position),
                0.0,
            )
        total_reward = float(np.sum(local_rewards))

        self.last_ball_x = ball_min_x
        self.frames += 1
        self.truncate = self.frames >= self.max_cycles
        self._episode_return += total_reward

        info = {
            "local_rewards": local_rewards,
            "total_reward": total_reward,
            "agent_rewards": local_rewards.copy(),
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), total_reward, bool(self.terminate), bool(self.truncate), info

    def render(self) -> np.ndarray | None:
        """Render the current state.

        Returns
        -------
        numpy.ndarray or None
            ``(screen_height, screen_width, 3)`` ``uint8`` frame in
            ``"rgb_array"`` mode; ``None`` otherwise.
        """
        if self.render_mode is None:
            if not self._warned_no_render_mode:
                warnings.warn(
                    "render() called with render_mode=None; create the environment with "
                    "render_mode='rgb_array' or 'human' to render.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_no_render_mode = True
            return None
        if self.ball is None:
            raise ResetNeededError("PistonballEnv.render() called before reset()")
        if self._renderer is None:
            from env_lib.pistonball_env.rendering import PistonballLayout, PistonballRenderer

            layout = PistonballLayout(
                n_pistons=self.n_pistons,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                wall_width=self.wall_width,
                piston_width=self.piston_width,
                piston_radius=self.piston_radius,
                piston_body_height=self.piston_body_height,
                ball_radius=self.ball_radius,
                minimum_piston_y=self.minimum_piston_y,
                maximum_piston_y=self.maximum_piston_y,
            )
            self._renderer = PistonballRenderer(
                layout, render_mode=self.render_mode, fps=self.metadata["render_fps"]
            )
        ball = self.ball
        return self._renderer.render(
            piston_y=self.piston_pos_y,
            observable=self.observable_mask(),
            ball_position=tuple(ball.position),
            ball_angle=float(ball.angle),
            ball_velocity=tuple(ball.velocity),
            step=self.frames,
            max_cycles=self.max_cycles,
            episode_return=self._episode_return,
            terminated=self.terminate,
            truncated=self.truncate,
        )

    def close(self) -> None:
        """Close the render window / release the renderer (idempotent)."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Observability and rewards
    # ------------------------------------------------------------------
    def _ball_piston_index(self, ball_x: float) -> int:
        """Index of the piston under ``ball_x`` (clipped to the valid range)."""
        index = int((ball_x - self.wall_width - self.piston_radius) / self.piston_width)
        return min(max(index, 0), self.n_pistons - 1)

    def _observer_slice(self, ball_x: float) -> slice:
        """Slice of the pistons within ``kappa`` hops of the piston under ``ball_x``."""
        index = self._ball_piston_index(ball_x)
        return slice(max(index - self.kappa, 0), index + self.kappa + 1)

    def observable_mask(self, ball_x: float | None = None) -> np.ndarray:
        """Boolean mask of the pistons that observe a ball at ``ball_x``.

        Parameters
        ----------
        ball_x:
            Ball x coordinate in pixels; defaults to the current ball centre.

        Returns
        -------
        numpy.ndarray
            ``bool`` array of shape ``(n_pistons,)``, ``True`` where
            ``|i - ball_piston_index| <= kappa``.
        """
        mask = np.zeros(self.n_pistons, dtype=bool)
        if ball_x is None:
            if self.ball is None:
                return mask
            ball_x = self.ball.position[0]
        mask[self._observer_slice(ball_x)] = True
        return mask

    def _can_observe_ball(self, piston_index: int, ball_x: float) -> bool:
        """Whether piston ``piston_index`` observes a ball at ``ball_x`` (kappa rule)."""
        if self.ball is None:
            return False
        return abs(piston_index - self._ball_piston_index(ball_x)) <= self.kappa

    def _local_rewards(self, prev_ball_x: int, curr_ball_x: int) -> np.ndarray:
        """Vectorised :meth:`get_local_reward_for_piston` for all pistons."""
        observed = np.zeros(self.n_pistons, dtype=bool)
        observed[self._observer_slice(prev_ball_x)] = True
        observed[self._observer_slice(curr_ball_x)] = True
        ball_reward = self.get_local_reward(prev_ball_x, curr_ball_x)
        return np.where(observed, ball_reward + self.time_penalty, self.time_penalty)

    def get_local_reward_for_piston(
        self, piston_index: int, prev_ball_x: float, curr_ball_x: float
    ) -> float:
        """Local reward of one piston (reference implementation of the reward rule).

        Parameters
        ----------
        piston_index:
            Piston index.
        prev_ball_x, curr_ball_x:
            Previous and current ball left-edge positions (``int(x - radius)``).

        Returns
        -------
        float
            ``time_penalty`` plus the ball term if the piston observes the ball
            at either position.
        """
        can_observe_prev = self._can_observe_ball(piston_index, prev_ball_x)
        can_observe_curr = self._can_observe_ball(piston_index, curr_ball_x)
        if not can_observe_prev and not can_observe_curr:
            return self.time_penalty
        return self.get_local_reward(prev_ball_x, curr_ball_x) + self.time_penalty

    def get_local_reward(self, prev_position: float, curr_position: float) -> float:
        """Ball term: ``0.5 * dx`` when the ball moved left by ``dx``, else ``-dx``."""
        if prev_position > curr_position:
            return 0.5 * (prev_position - curr_position)
        return prev_position - curr_position

    @property
    def last_ball_positions(self) -> np.ndarray | None:
        """Previous ball left-edge position repeated for every piston (legacy view)."""
        if self.last_ball_x is None:
            return None
        return np.full(self.n_pistons, self.last_ball_x)

    def _get_obs(self) -> np.ndarray:
        """Joint observation, ``float32`` array of shape ``(n_pistons, 7)``."""
        obs = self._obs_template.copy()
        obs[:, 0] = (self.piston_pos_y - self.mid_piston_y) / self.piston_y_half_range
        if self.ball is None:
            return obs
        x, y = self.ball.position
        vx, vy = self.ball.velocity
        arena_width, arena_height = self._arena_size
        obs[self._observer_slice(x), 2:] = (
            (x - self.wall_width) / arena_width,
            (y - self.wall_width) / arena_height,
            vx / 15,
            vy / 8,
            self.ball.angular_velocity / 8,
        )
        return obs

    def _action_to_displacement(self, action: Any) -> np.ndarray:
        """Validate ``action`` and convert it to piston displacements in ``[-1, 1]``."""
        if self.continuous:
            try:
                values = np.asarray(action, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid continuous action {action!r}: {exc}") from None
            self._check_action_shape(values)
            if not np.all(np.isfinite(values)):
                raise ValueError("Continuous actions must be finite (got NaN or inf)")
            return np.clip(values, -1.0, 1.0).astype(np.float64)
        values = np.asarray(action)
        self._check_action_shape(values)
        if values.dtype.kind not in "biuf":
            raise ValueError(f"Discrete actions must be numeric, got dtype {values.dtype}")
        if values.dtype.kind == "f" and not np.all(
            np.isfinite(values) & (values == np.round(values))
        ):
            raise ValueError(f"Discrete actions must be integers in {{0, 1, 2}}, got {values}")
        if np.any((values < 0) | (values > 2)):
            raise ValueError(
                f"Discrete actions must be in {{0, 1, 2}} (down, stay, up), got {values}"
            )
        return values.astype(np.float64) - 1.0

    def _check_action_shape(self, values: np.ndarray) -> None:
        if values.shape != (self.n_pistons,):
            raise ValueError(
                f"Action shape {values.shape} does not match expected shape ({self.n_pistons},)"
            )

    # ------------------------------------------------------------------
    # Deprecated drawing API (rendering now lives in PistonballRenderer)
    # ------------------------------------------------------------------
    def _warn_drawing_api(self, name: str) -> None:
        warnings.warn(
            f"PistonballEnv.{name}() is deprecated and does nothing; drawing happens inside "
            "render().",
            DeprecationWarning,
            stacklevel=3,
        )

    def draw(self) -> None:
        """Deprecated no-op; use :meth:`render`."""
        self._warn_drawing_api("draw")

    def draw_background(self) -> None:
        """Deprecated no-op; use :meth:`render`."""
        self._warn_drawing_api("draw_background")

    def draw_pistons(self, observable_pistons: Sequence[int] | None = None) -> None:
        """Deprecated no-op; use :meth:`render`."""
        self._warn_drawing_api("draw_pistons")

    def enable_render(self) -> None:
        """Deprecated no-op; :meth:`render` opens the window on demand."""
        self._warn_drawing_api("enable_render")

    @property
    def renderer(self) -> Any:
        """The :class:`~env_lib.pistonball_env.rendering.PistonballRenderer`, or ``None``."""
        return self._renderer
