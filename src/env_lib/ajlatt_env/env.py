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
from env_lib.utils.rendering import validate_render_mode

__all__ = ["AJLATTEnv"]


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

    def __init__(
        self,
        config: AJLATTConfig | None = None,
        render_mode: str | None = None,
        **overrides: Any,
    ):
        super().__init__()
        validate_render_mode(render_mode)
        if config is None:
            config = AJLATTConfig.from_kwargs(**overrides)
        elif overrides:
            config = AJLATTConfig.from_kwargs(**{**config.to_dict(), **overrides})
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
            raise RuntimeError("AJLATTEnv.step() called before reset()")
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
            "agent_rewards": np.asarray(reward, dtype=np.float64),
            "team_reward": float(np.sum(reward)),
            "collisions": np.asarray(collided, dtype=bool),
            "episode_collisions": self._episode_collisions.copy(),
            "target_observed": self.RT_obs[:, self.nR].astype(bool),
            "communication": self.com_plot.astype(bool),
            "robot_cov_trace": np.array([np.trace(est.cov) for est in self.robot_est]),
            "target_cov_trace": np.array(
                [np.trace(self.target_est[i][0].cov) for i in range(self.nR)]
            ),
            "step": self.step_count,
        }

    def _record_history(self, reward) -> None:
        robot_true = np.array([agent.state[:2] for agent in self.robot_true])
        robot_est = np.array([est.state[:2] for est in self.robot_est])
        target_est = np.array([self.target_est[i][0].state[:2] for i in range(self.nR)])
        self._history["robot_cov_trace"].append(
            np.array([np.trace(est.cov) for est in self.robot_est])
        )
        self._history["target_cov_trace"].append(
            np.array([np.trace(self.target_est[i][0].cov) for i in range(self.nR)])
        )
        self._history["robot_error"].append(np.linalg.norm(robot_est - robot_true, axis=1))
        self._history["target_error"].append(
            np.linalg.norm(target_est - self.target_true[0].state[:2], axis=1)
        )
        self._history["reward"].append(np.asarray(reward, dtype=float).copy())

    def _observation(self, com_obs: np.ndarray) -> np.ndarray:
        """Stack the per-robot observations (all quantities expressed in the robot frame)."""
        obs = np.zeros((self.nR, self.obs_dim))
        poses = np.array([est.state for est in self.robot_est])
        cov_traces = np.array([np.trace(est.cov) for est in self.robot_est])
        cfg = self.config
        lo = self.MAP.mapmin + cfg.obstacle_sensing_margin
        hi = self.MAP.mapmax - cfg.obstacle_sensing_margin
        for i in range(self.nR):
            theta = poses[i, 2]
            c, s = np.cos(theta), np.sin(theta)
            rot_t = np.array([[c, s], [-s, c]])  # C^T
            target = self.target_est[i][0]
            obs[i, 0:2] = rot_t @ (target.state[:2] - poses[i, :2])
            obs[i, 2] = target.state[2] - theta
            obs[i, 3:5] = self.target_velocity
            obs[i, 5] = np.trace(target.cov)

            others = [j for j in range(self.nR) if j != i]
            if others:
                block = obs[i, 6 : 6 + 4 * len(others)].reshape(len(others), 4)
                block[:, 0:2] = (poses[others, :2] - poses[i, :2]) @ rot_t.T
                block[:, 2] = poses[others, 2] - theta
                block[:, 3] = cov_traces[others]

            if lo[0] <= poses[i, 0] <= hi[0] and lo[1] <= poses[i, 1] <= hi[1]:
                closest = self.MAP.get_closest_obstacle(
                    poses[i], fov=self._fov_rad, r_max=cfg.sensor_r_max
                )
                obs[i, -6:-4] = (cfg.sensor_r_max, np.pi) if closest is None else closest
            obs[i, -4:-1] = poses[i]
            obs[i, -1] = cov_traces[i]
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
        for det_id in range(self.nR):
            self._localise(det_id, zr, R)
            self._track_targets(det_id, zr, R, com_obs, rt_obs)

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
            s_list.append(h_self.T @ inv_r @ h_self)
            y_list.append(h_self.T @ inv_r @ z_bar)
        if not s_list:
            return  # nothing observed: the fused estimate equals the prior
        omega = psd_inverse(me.cov)
        s_list.append(omega)
        y_list.append(omega @ me.state)
        me.cov, me.state = covariance_intersection(
            np.stack(s_list), np.stack(y_list, axis=1), solver=self.config.ci_solver
        )

    def _track_targets(self, det_id: int, zr, R, com_obs, rt_obs) -> None:
        """Update robot ``det_id``'s target beliefs from its neighbourhood."""
        nR = self.nR
        solver = self.config.ci_solver
        neighbours = np.flatnonzero(com_obs[det_id] == 1)
        for t in range(self.nT):
            dim = 3 if t == 0 else 2
            detectors = [int(r) for r in neighbours if rt_obs[r, nR + t] == 1]

            omega = np.zeros((len(neighbours), dim, dim))
            q = np.zeros((dim, len(neighbours)))
            for k, r in enumerate(neighbours):
                prior = self.target_est[r][t]
                omega[k] = psd_inverse(prior.cov)
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
        return h_target.T @ inv_r @ h_target, h_target.T @ inv_r @ z_bar

    def get_reward(self, rt_obs) -> tuple[np.ndarray, np.ndarray]:
        """Per-robot rewards and obstacle-collision flags."""
        cfg = self.config
        nR = self.nR
        reward = np.array(
            [
                -cfg.target_cov_weight * np.trace(self.target_est[i][0].cov)
                - cfg.robot_cov_weight * np.trace(self.robot_est[i].cov)
                for i in range(nR)
            ]
        )
        collided = np.zeros(nR, dtype=bool)
        poses = np.array([est.state for est in self.robot_est])
        lo, hi = self.MAP.mapmin + 0.1, self.MAP.mapmax - 0.1
        for i in range(nR):
            if np.any(poses[i, :2] < lo) or np.any(poses[i, :2] > hi):
                reward[i] -= cfg.boundary_penalty
                self._episode_collisions[i] += 1
                continue
            closest = self.MAP.get_closest_obstacle(poses[i], fov=2 * np.pi, r_max=cfg.sensor_r_max)
            if closest is not None and closest[0] < cfg.obstacle_collision_distance:
                reward[i] -= cfg.obstacle_penalty
                self._episode_collisions[i] += 1
                collided[i] = True

        distance = np.linalg.norm(poses[:, None, :2] - poses[None, :, :2], axis=-1)
        np.fill_diagonal(distance, np.inf)
        close = np.any(distance < cfg.mutual_collision_distance, axis=1)
        reward[close] -= cfg.mutual_collision_penalty
        self._episode_collisions[close] += 1
        return reward, collided
