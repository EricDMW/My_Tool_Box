"""AJLATT: active joint localisation and target tracking with a robot team.

A team of unicycle robots with range-bearing sensors localises itself and
tracks a moving target on an occupancy grid map. Each robot runs an estimator
for its own pose and for the target; estimates are fused over a limited
communication range with covariance intersection. The reward drives the team
to keep the target (and its own pose) well estimated while avoiding obstacles,
the map border and each other.

API
---
The environment follows the Gymnasium API with joint multi-agent spaces:

* ``action``: array ``(num_robots, 2)`` of ``(v, omega)`` commands.
* ``observation``: array ``(num_robots, obs_dim)``; see
  :meth:`AJLATTEnv.observation_layout`.
* ``reward``: array ``(num_robots,)`` of per-robot rewards (the team reward is
  their sum; ``info["team_reward"]``).
* ``terminated``: array ``(num_robots,)`` of per-robot collision flags
  (``any(terminated)`` if a single flag is needed).
* ``truncated``: ``bool``, set after ``max_episode_steps`` steps.

Use :class:`env_lib.ajlatt_env.TeamRewardWrapper` for a scalar-reward,
scalar-termination view.
"""

from __future__ import annotations

import warnings
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env_lib.ajlatt_env.config import AJLATTConfig
from env_lib.ajlatt_env.controllers import TargetPolicy, make_target_policy
from env_lib.ajlatt_env.estimation import (
    AgentEstimate,
    AgentState,
    covariance_intersection,
    measurement_model,
    pi_to_pi,
    psd_inverse,
)
from env_lib.ajlatt_env.maps import DynamicMap, GridMap, load_grid_map
from env_lib.errors import ResetNeededError
from env_lib.utils.rendering import state_without_renderer, validate_render_mode

__all__ = ["AJLATTEnv", "make"]


class AJLATTEnv(gym.Env):
    """Multi-robot active localisation and target tracking.

    Parameters
    ----------
    config:
        Full configuration; defaults to :class:`AJLATTConfig()`.
    render_mode:
        ``None``, ``"human"`` or ``"rgb_array"``.
    **overrides:
        Individual config fields (e.g. ``map_name="obstacles05"``,
        ``num_robots=3``) applied on top of ``config``.

    Examples
    --------
    >>> import numpy as np
    >>> from env_lib import AJLATTEnv
    >>> env = AJLATTEnv(map_name="obstacles04", num_robots=4)
    >>> obs, info = env.reset(seed=0)
    >>> obs, reward, terminated, truncated, info = env.step(np.tile([0.2, 0.0], (4, 1)))
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __getstate__(self) -> dict[str, Any]:
        # Pickle and deep-copy without the renderer (rebuilt on the next render()).
        return state_without_renderer(self)

    def __init__(
        self,
        config: AJLATTConfig | None = None,
        render_mode: str | None = None,
        **overrides: Any,
    ):
        super().__init__()
        legacy_render = overrides.pop("render", None)
        if legacy_render is not None:
            warnings.warn(
                "The 'render' flag is deprecated; pass render_mode='human' or 'rgb_array'",
                DeprecationWarning,
                stacklevel=2,
            )
            if render_mode is None and legacy_render:
                render_mode = "human"
        validate_render_mode(render_mode)
        if config is None:
            config = AJLATTConfig.from_kwargs(_stacklevel=3, **overrides)
        elif overrides:
            config = AJLATTConfig.from_kwargs(_stacklevel=3, **{**config.to_dict(), **overrides})
        self.config = config
        self.render_mode = render_mode

        self.nR = self.num_robots = config.num_robots
        self.nT = self.num_targets = config.num_targets
        self.dt = config.sampling_period
        self.obs_dim = self.state_num = config.observation_dim
        self.action_dim = 2

        self.MAP: GridMap = load_grid_map(config.map_name, margin2wall=config.margin2wall)
        self._target_policy: TargetPolicy = make_target_policy(
            config.target_policy,
            config.map_name,
            velocity=config.target_linear_velocity,
            k_theta=config.k_theta,
            omega_max=config.target_omega_bound,
        )

        low = np.array([0.0, -config.max_angular_velocity], dtype=np.float32)
        high = np.array([config.max_linear_velocity, config.max_angular_velocity], dtype=np.float32)
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(
            np.tile(low, (self.nR, 1)), np.tile(high, (self.nR, 1)), dtype=np.float32
        )
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.nR, self.obs_dim), np.float32)

        # Initial conditions.
        self._robot_init_pose = config.robot_poses()
        self._target_init_pose = config.target_poses()
        self._robot_init_cov = np.zeros((self.nR, 3, 3))
        for i in range(self.nR):
            self._robot_init_cov[i] = config.robot_init_cov[i] * np.eye(3)
            self._robot_init_cov[i, 2, 2] = 1e-3
        self._target_init_cov = np.zeros((self.nT, 3, 3))
        for j in range(self.nT):
            self._target_init_cov[j] = config.target_init_cov[j] * np.eye(3)
            self._target_init_cov[j, 2, 2] = 1e-1

        # Agents: ground truth and beliefs (each robot keeps its own target beliefs).
        self.robot_true = [AgentState(3, self.dt) for _ in range(self.nR)]
        self.robot_est = [AgentEstimate(3, self.dt) for _ in range(self.nR)]
        self.target_true = [AgentState(3, self.dt) for _ in range(self.nT)]
        self.target_est = [
            [AgentEstimate(3 if j == 0 else 2, self.dt) for j in range(self.nT)]
            for _ in range(self.nR)
        ]

        self.com_plot = np.eye(self.nR)
        self.RT_obs = np.zeros((self.nR, self.nR + self.nT), dtype=int)
        self.target_velocity = np.zeros(2)
        self.step_count = 0
        self._needs_reset = True
        self._renderer = None
        self._warned_extra_rows = False
        self._history: dict[str, list[np.ndarray]] = {}
        self._fov_rad = np.deg2rad(config.fov)
        self._obstacle_cache: tuple | None = None
        self._prior_information: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Indices of the other robots, row i = [j for j != i] (observation layout order).
        self._others = np.array(
            [[j for j in range(self.nR) if j != i] for i in range(self.nR)], dtype=np.intp
        ).reshape(self.nR, self.nR - 1)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if (
            seed is None
            and self._needs_reset
            and self.config.seed is not None
            and self._np_random is None
        ):
            seed = self.config.seed
        super().reset(seed=seed)
        rng = self.np_random

        self._obstacle_cache = None
        if isinstance(self.MAP, DynamicMap):
            self.MAP.generate_map(rng=rng)
        self._target_policy.reset(rng)

        for j in range(self.nT):
            self.target_true[j].reset(self._target_init_pose[j])
        for i in range(self.nR):
            self.robot_true[i].reset(self._robot_init_pose[i])
            self.robot_est[i].reset(self._robot_init_pose[i], self._robot_init_cov[i], rng)
            for j in range(self.nT):
                if j == 0:
                    self.target_est[i][j].reset(
                        self._target_init_pose[j], self._target_init_cov[j], rng
                    )
                else:
                    self.target_est[i][j].reset(
                        self._target_init_pose[j, :2], self._target_init_cov[j, :2, :2], rng
                    )

        self.step_count = 0
        self._needs_reset = False
        self.com_plot = np.eye(self.nR)
        self.RT_obs = np.zeros((self.nR, self.nR + self.nT), dtype=int)
        self.target_velocity = np.zeros(2)
        self._episode_collisions = np.zeros(self.nR, dtype=int)
        self._history = {
            key: []
            for key in (
                "robot_cov_trace",
                "target_cov_trace",
                "robot_error",
                "target_error",
                "reward",
            )
        }
        self._record_history(np.zeros(self.nR))
        if self._renderer is not None:
            self._renderer.reset()

        observation = self._observation(np.eye(self.nR))
        info = self._info(np.zeros(self.nR), np.zeros(self.nR, dtype=bool))
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        if self._needs_reset:
            raise ResetNeededError("AJLATTEnv.step() called before reset()")
        actions = self._validate_action(action)
        cfg = self.config
        rng = self.np_random

        # Robot motion: beliefs use the commanded input, ground truth a noisy one.
        for i in range(self.nR):
            v, w = actions[i]
            if cfg.process_noise_fixed:
                sigma_v, sigma_w = cfg.sigma_vR, cfg.sigma_wR
            else:
                sigma = cfg.sigmapR * v
                sigma_v, sigma_w = sigma / np.sqrt(2), 2 * np.sqrt(2) * sigma
            self.robot_est[i].propagate((v, w), sigma_v, sigma_w)
            self.robot_true[i].propagate((rng.normal(v, sigma_v), rng.normal(w, sigma_w)))

        # Target motion (only target 0 moves) and belief prediction.
        v_t, w_t = self._target_policy(self.target_true[0].state)
        self.target_velocity = np.array([v_t, w_t], dtype=float)
        self.target_true[0].propagate((v_t, w_t))
        if cfg.process_noise_fixed:
            sigma_vT, sigma_wT = cfg.sigma_vT, cfg.sigma_wT
        else:
            sigma = cfg.sigmapT * cfg.target_linear_velocity
            sigma_vT, sigma_wT = sigma / np.sqrt(2), 2 * np.sqrt(2) * sigma
        noisy = (rng.normal(v_t, sigma_vT), rng.normal(w_t, sigma_wT))
        for i in range(self.nR):
            self.target_est[i][0].propagate(noisy, sigma_vT, sigma_wT)

        # Measurement update and fusion.
        if cfg.use_update:
            zr, R, com_obs, rt_obs = self.measurement_generation()
            self.update(zr, R, com_obs, rt_obs)
        else:
            com_obs = np.eye(self.nR)
            rt_obs = np.zeros((self.nR, self.nR + self.nT), dtype=int)
        self.com_plot = com_obs
        self.RT_obs = rt_obs

        reward, collided = self.get_reward(rt_obs)
        numerical_error = not all(np.isfinite(est.cov).all() for est in self.robot_est)
        if numerical_error:
            warnings.warn(
                "AJLATT estimator diverged (non-finite covariance); terminating episode",
                RuntimeWarning,
                stacklevel=2,
            )
        terminated = (
            collided.copy() if cfg.terminate_on_collision else np.zeros(self.nR, dtype=bool)
        )
        if numerical_error:
            terminated[:] = True

        self.step_count += 1
        truncated = self.step_count >= cfg.max_episode_steps
        self._record_history(reward)
        observation = self._observation(com_obs)
        info = self._info(reward, collided)
        info["numerical_error"] = numerical_error
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, bool(truncated), info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "render() called without a render_mode; pass render_mode='rgb_array' or 'human'"
            )
            return None
        if self._needs_reset:
            raise ResetNeededError("AJLATTEnv.render() called before reset()")
        if self._renderer is None:
            from env_lib.ajlatt_env.rendering import AJLATTRenderer

            self._renderer = AJLATTRenderer(self, self.render_mode, fps=self.metadata["render_fps"])
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Helpers exposed for analysis
    # ------------------------------------------------------------------
    def observation_layout(self) -> dict[str, slice]:
        """Slices of the per-robot observation vector."""
        n = 6 + 4 * (self.nR - 1)
        return {
            "target_relative_position": slice(0, 2),
            "target_relative_heading": slice(2, 3),
            "target_velocity": slice(3, 5),
            "target_cov_trace": slice(5, 6),
            "neighbours": slice(6, n),  # per neighbour: rel. x, rel. y, rel. heading, cov trace
            "closest_obstacle": slice(n, n + 2),  # range, bearing
            "self_pose": slice(n + 2, n + 5),
            "self_cov_trace": slice(n + 5, n + 6),
        }

    def episode_statistics(self) -> dict[str, np.ndarray]:
        """Per-step traces of the current episode, each of shape ``(steps + 1, num_robots)``.

        Keys: ``robot_cov_trace``, ``target_cov_trace`` (trace of the self and
        target-0 covariances), ``robot_error``, ``target_error`` (Euclidean
        position errors of the beliefs) and ``reward``.
        """
        return {key: np.asarray(values) for key, values in self._history.items()}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _validate_action(self, action) -> np.ndarray:
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        actions = np.asarray(action, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 2 or actions.shape[0] < self.nR:
            raise ValueError(f"action must have shape ({self.nR}, 2), got {actions.shape}")
        if not np.all(np.isfinite(actions[: self.nR])):
            raise ValueError("action contains NaN or infinite values")
        if actions.shape[0] > self.nR:
            if not self._warned_extra_rows:
                warnings.warn(
                    f"action has {actions.shape[0]} rows for {self.nR} robots; extra rows are ignored",
                    stacklevel=3,
                )
                self._warned_extra_rows = True
            actions = actions[: self.nR]
        if self.config.clip_actions:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions

    def _info(self, reward: np.ndarray, collided: np.ndarray) -> dict[str, Any]:
        return {
            "agent_rewards": np.array(reward, dtype=np.float64),
            "team_reward": float(np.sum(reward)),
            "collisions": np.asarray(collided, dtype=bool),
            "episode_collisions": self._episode_collisions.copy(),
            "target_observed": self.RT_obs[:, self.nR].astype(bool),
            "communication": self.com_plot.astype(bool),
            **dict(zip(("robot_cov_trace", "target_cov_trace"), self._cov_traces())),
            "step": self.step_count,
        }

    def _obstacle_fans(self, poses: np.ndarray) -> tuple[list, list]:
        """Closest obstacle of every robot in the 360-degree fan and in the sensor fan.

        ``get_reward`` (collision check, full circle) and ``_observation``
        (sensor field of view) query the same estimated poses within a step,
        so both fans are cast in one batch and cached for these poses. The
        results equal per-robot :meth:`GridMap.get_closest_obstacle` calls.
        """
        key = (id(self.MAP), poses.tobytes())
        if self._obstacle_cache is not None and self._obstacle_cache[0] == key:
            return self._obstacle_cache[1]
        full, sensor = [None] * self.nR, [None] * self.nR
        finite = np.flatnonzero(np.isfinite(poses).all(axis=1))
        if finite.size:
            fans = self.MAP.closest_obstacles(
                poses[finite], fov=[2 * np.pi, self._fov_rad], r_max=self.config.sensor_r_max
            )
            for k, i in enumerate(finite):
                full[i], sensor[i] = fans[0][k], fans[1][k]
        self._obstacle_cache = (key, (full, sensor))
        return full, sensor

    def _cov_traces(self) -> tuple[np.ndarray, np.ndarray]:
        """Traces of the robots' self covariances and of their target-0 covariances."""
        robot = np.array([est.cov for est in self.robot_est])
        target = np.array([self.target_est[i][0].cov for i in range(self.nR)])
        return np.trace(robot, axis1=1, axis2=2), np.trace(target, axis1=1, axis2=2)

    def _record_history(self, reward) -> None:
        robot_true = np.array([agent.state[:2] for agent in self.robot_true])
        robot_est = np.array([est.state[:2] for est in self.robot_est])
        target_est = np.array([self.target_est[i][0].state[:2] for i in range(self.nR)])
        robot_traces, target_traces = self._cov_traces()
        self._history["robot_cov_trace"].append(robot_traces)
        self._history["target_cov_trace"].append(target_traces)
        self._history["robot_error"].append(np.linalg.norm(robot_est - robot_true, axis=1))
        self._history["target_error"].append(
            np.linalg.norm(target_est - self.target_true[0].state[:2], axis=1)
        )
        self._history["reward"].append(np.asarray(reward, dtype=float).copy())

    def _observation(self, com_obs: np.ndarray) -> np.ndarray:
        """Stack the per-robot observations (all quantities expressed in the robot frame)."""
        nR = self.nR
        cfg = self.config
        obs = np.zeros((nR, self.obs_dim))
        poses = np.array([est.state for est in self.robot_est])
        cov_traces, target_traces = self._cov_traces()
        targets = [self.target_est[i][0] for i in range(nR)]
        theta = poses[:, 2]
        cos, sin = np.cos(theta), np.sin(theta)
        rot_t = np.empty((nR, 2, 2))  # C^T of every robot
        rot_t[:, 0, 0] = cos
        rot_t[:, 0, 1] = sin
        rot_t[:, 1, 0] = -sin
        rot_t[:, 1, 1] = cos
        target_states = np.array([target.state for target in targets])
        offset = (target_states[:, :2] - poses[:, :2])[:, :, np.newaxis]
        obs[:, 0:2] = (rot_t @ offset)[:, :, 0]
        obs[:, 2] = target_states[:, 2] - theta
        obs[:, 3:5] = self.target_velocity
        obs[:, 5] = target_traces
        if nR > 1:
            others = self._others
            block = obs[:, 6 : 6 + 4 * (nR - 1)].reshape(nR, nR - 1, 4)
            block[:, :, 0:2] = (poses[others, :2] - poses[:, np.newaxis, :2]) @ rot_t.transpose(
                0, 2, 1
            )
            block[:, :, 2] = poses[others, 2] - theta[:, np.newaxis]
            block[:, :, 3] = cov_traces[others]

        lo = self.MAP.mapmin + cfg.obstacle_sensing_margin
        hi = self.MAP.mapmax - cfg.obstacle_sensing_margin
        sensing = np.flatnonzero(np.all((lo <= poses[:, :2]) & (poses[:, :2] <= hi), axis=1))
        if sensing.size:
            _, closest = self._obstacle_fans(poses)
            for i in sensing:
                obs[i, -6:-4] = (cfg.sensor_r_max, np.pi) if closest[i] is None else closest[i]
        obs[:, -4:-1] = poses
        obs[:, -1] = cov_traces
        return obs.astype(np.float32)

    def measurement_generation(self):
        """Simulate range-bearing measurements between robots and towards targets.

        Returns
        -------
        zr:
            ``zr[l][j]`` is the noisy ``(range, bearing)`` robot ``l`` measured of
            entity ``j`` (robots first, then targets), or ``None``.
        R:
            Matching measurement noise covariances (``2x2``) or ``None``.
        com_obs:
            ``(nR, nR)`` communication adjacency (includes self-loops).
        rt_obs:
            ``(nR, nR + nT)`` detection matrix.
        """
        cfg = self.config
        nR, nT = self.nR, self.nT
        rng = self.np_random
        observer = np.array([agent.state for agent in self.robot_true])
        entities = np.vstack([observer, np.array([agent.state for agent in self.target_true])])
        est_xy = np.array([est.state[:2] for est in self.robot_est])

        delta = entities[None, :, :2] - observer[:, None, :2]  # (nR, nR+nT, 2)
        rng_ = np.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2)
        bearing = pi_to_pi(np.arctan2(delta[..., 1], delta[..., 0]) - observer[:, None, 2])

        com_obs = np.eye(nR)
        within = rng_[:, :nR] < cfg.commu_r_max
        np.fill_diagonal(within, False)
        com_obs[within] = 1

        inside_map = np.all((est_xy >= self.MAP.mapmin) & (est_xy <= self.MAP.mapmax), axis=1)
        candidate = (
            (rng_ < cfg.sensor_r_max)
            & (rng_ > cfg.sensor_r_min)
            & (np.abs(bearing) <= self._fov_rad / 2)
            & inside_map[:, None]
        )
        candidate[np.arange(nR), np.arange(nR)] = False
        rows, cols = np.nonzero(candidate)
        if rows.size:
            blocked = self.MAP.is_blocked_batch(observer[rows], entities[cols])
            candidate[rows[blocked], cols[blocked]] = False
        rt_obs = candidate.astype(int)

        zr: list[list[np.ndarray | None]] = [[None] * (nR + nT) for _ in range(nR)]
        R: list[list[np.ndarray | None]] = [[None] * (nR + nT) for _ in range(nR)]
        for ell, j in zip(*np.nonzero(candidate)):
            r = rng_[ell, j]
            sigma_r = cfg.sigma_p * r if cfg.range_noise_proportional else cfg.sigma_r
            zr[ell][j] = np.array(
                [rng.normal(r, sigma_r), rng.normal(bearing[ell, j], cfg.sigma_th)]
            )
            R[ell][j] = np.diag([sigma_r**2, cfg.sigma_th**2])
        return zr, R, com_obs, rt_obs

    def update(self, zr, R, com_obs, rt_obs) -> None:
        """Measurement update, processed robot by robot.

        For each robot in index order, first its self-localisation is fused
        (covariance intersection of its relative measurements with its prior),
        then its target beliefs are updated from the measurements and priors of
        its communication neighbours. Updates are applied in place, so later
        robots use the already updated beliefs of earlier ones (as in the
        original implementation).
        """
        # Inverses of target priors, shared by the robots of a neighbourhood. Keyed
        # by the covariance array itself: estimates are replaced (never modified
        # in place) when updated, and the cache keeps the arrays alive.
        self._prior_information = {}
        try:
            for det_id in range(self.nR):
                self._localise(det_id, zr, R)
                self._track_targets(det_id, zr, R, com_obs, rt_obs)
        finally:
            self._prior_information = {}

    def _localise(self, det_id: int, zr, R) -> None:
        """Self-localisation of robot ``det_id`` from its relative measurements."""
        nR, nT = self.nR, self.nT
        me = self.robot_est[det_id]
        s_list, y_list = [], []
        for other in range(nR + nT):
            z = zr[det_id][other]
            if z is None:
                continue
            if other < nR:
                obj = self.robot_est[other]
                zhat, h_self, h_obj = measurement_model(me.state, obj.state)
            else:
                obj = self.target_est[det_id][other - nR]
                zhat, h_self, h_obj = measurement_model(me.state, obj.state)
                if other - nR != 0:
                    h_obj = h_obj[:, :2]
            residual = np.array([z[0] - zhat[0], pi_to_pi(z[1] - zhat[1])])
            inv_r = psd_inverse(R[det_id][other] + h_obj @ obj.cov @ h_obj.T)
            z_bar = residual + h_self @ me.state
            weighted = h_self.T @ inv_r
            s_list.append(weighted @ h_self)
            y_list.append(weighted @ z_bar)
        if not s_list:
            return  # nothing observed: the fused estimate equals the prior
        omega = psd_inverse(me.cov)
        s_list.append(omega)
        y_list.append(omega @ me.state)
        me.cov, me.state = covariance_intersection(
            np.array(s_list), np.stack(y_list, axis=1), solver=self.config.ci_solver
        )

    def _track_targets(self, det_id: int, zr, R, com_obs, rt_obs) -> None:
        """Update robot ``det_id``'s target beliefs from its neighbourhood."""
        nR = self.nR
        solver = self.config.ci_solver
        neighbours = np.flatnonzero(com_obs[det_id] == 1)
        for t in range(self.nT):
            dim = 3 if t == 0 else 2
            detectors = neighbours[rt_obs[neighbours, nR + t] == 1].tolist()

            omega = np.zeros((len(neighbours), dim, dim))
            q = np.zeros((dim, len(neighbours)))
            for k, r in enumerate(neighbours):
                prior = self.target_est[r][t]
                cached = self._prior_information.get(id(prior.cov))
                if cached is None or cached[0] is not prior.cov:
                    cached = (prior.cov, psd_inverse(prior.cov))
                    self._prior_information[id(prior.cov)] = cached
                omega[k] = cached[1]
                q[:, k] = omega[k] @ prior.state

            if detectors:
                s = np.zeros((len(detectors), dim, dim))
                y = np.zeros((dim, len(detectors)))
                for k, r in enumerate(detectors):
                    s[k], y[:, k] = self._target_information(zr, R, r, t)
                s_est, y_est = covariance_intersection(s, y, information_form=True, solver=solver)
                s_prior, y_prior = covariance_intersection(
                    omega, q, information_form=True, solver=solver
                )
                cov, mean = covariance_intersection(
                    np.stack([s_est, s_prior]), np.stack([y_est, y_prior], axis=1), solver=solver
                )
            else:
                cov, mean = covariance_intersection(omega, q, solver=solver)
            estimate = self.target_est[det_id][t]
            estimate.state, estimate.cov = mean, cov

    def _target_information(self, zr, R, robot: int, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Information pair contributed by ``robot``'s measurement of target ``t``."""
        target = self.target_est[robot][t]
        me = self.robot_est[robot]
        z = zr[robot][self.nR + t]
        zhat, h_robot, h_target = measurement_model(me.state, target.state)
        if t != 0:
            h_target = h_target[:, :2]
        residual = np.array([z[0] - zhat[0], pi_to_pi(z[1] - zhat[1])])
        inv_r = psd_inverse(R[robot][self.nR + t] + h_robot @ me.cov @ h_robot.T)
        z_bar = residual + h_target @ target.state
        weighted = h_target.T @ inv_r
        return weighted @ h_target, weighted @ z_bar

    def get_reward(self, rt_obs) -> tuple[np.ndarray, np.ndarray]:
        """Per-robot rewards and obstacle-collision flags."""
        cfg = self.config
        nR = self.nR
        robot_traces, target_traces = self._cov_traces()
        reward = -cfg.target_cov_weight * target_traces - cfg.robot_cov_weight * robot_traces
        collided = np.zeros(nR, dtype=bool)
        poses = np.array([est.state for est in self.robot_est])
        lo, hi = self.MAP.mapmin + 0.1, self.MAP.mapmax - 0.1
        outside = np.any(poses[:, :2] < lo, axis=1) | np.any(poses[:, :2] > hi, axis=1)
        closest, _ = self._obstacle_fans(poses)
        for i in range(nR):
            if outside[i]:
                reward[i] -= cfg.boundary_penalty
                self._episode_collisions[i] += 1
                continue
            nearest = closest[i]
            if nearest is not None and nearest[0] < cfg.obstacle_collision_distance:
                reward[i] -= cfg.obstacle_penalty
                self._episode_collisions[i] += 1
                collided[i] = True

        distance = np.linalg.norm(poses[:, None, :2] - poses[None, :, :2], axis=-1)
        np.fill_diagonal(distance, np.inf)
        close = np.any(distance < cfg.mutual_collision_distance, axis=1)
        reward[close] -= cfg.mutual_collision_penalty
        self._episode_collisions[close] += 1
        return reward, collided


def make(*args: Any, **kwargs: Any) -> AJLATTEnv:
    """Deprecated location of :func:`env_lib.ajlatt_env.make` (kept for old imports)."""
    from env_lib.ajlatt_env import make as _make

    return _make(*args, **kwargs)
