"""Tests for the AJLATT environment, its configuration and estimation core."""

import argparse
import warnings

import gymnasium as gym
import numpy as np
import pytest

import env_lib
from env_lib.ajlatt_env import AJLATTConfig, AJLATTEnv, TeamRewardWrapper, make
from env_lib.ajlatt_env.controllers import (
    CirclePolicy,
    SinePolicy,
    WaypointPolicy,
    make_target_policy,
)
from env_lib.ajlatt_env.estimation import (
    AgentEstimate,
    ci_weights,
    covariance_intersection,
    measurement_model,
    pi_to_pi,
    psd_inverse,
    se2_step,
)
from env_lib.utils import figure_to_rgb, set_theme


def _forward(n=4, v=0.2, w=0.0):
    return np.tile([v, w], (n, 1))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_config_defaults_and_dimensions():
    cfg = AJLATTConfig()
    assert cfg.num_robots == 4 and cfg.max_episode_steps == 120
    assert cfg.observation_dim == 7 + 4 * 3 + 5
    assert cfg.robot_poses().shape == (4, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_robots": 0},
        {"num_robots": 7},  # only six default poses
        {"sensor_r_max": 0.0},
        {"fov": 0.0},
        {"ci_solver": "bogus"},
        {"sigma_th": -1.0},
        {"num_targets": 12},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        AJLATTConfig(**kwargs)


def test_config_legacy_names():
    with pytest.warns(DeprecationWarning, match="num_robots"):
        cfg = AJLATTConfig.from_kwargs(num_Robot=3)
    assert cfg.num_robots == 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = AJLATTConfig.from_kwargs(
            T_steps=80, maxmum_run=40, SIGPERCENT=0, useupdate=0, figID=2
        )
    assert cfg.max_episode_steps == 41
    assert cfg.range_noise_proportional is False and cfg.use_update is False
    with pytest.raises(TypeError):
        AJLATTConfig.from_kwargs(not_a_parameter=1)


def test_config_argparse_round_trip():
    parser = AJLATTConfig.add_arguments(argparse.ArgumentParser())
    ns = parser.parse_args(
        ["--num_robots", "3", "--use_update", "false", "--map_name", "obstacles05"]
    )
    cfg = AJLATTConfig.from_namespace(ns)
    assert (cfg.num_robots, cfg.use_update, cfg.map_name) == (3, False, "obstacles05")
    assert AJLATTConfig.from_dict(cfg.to_dict()) == cfg


# ---------------------------------------------------------------------------
# Environment API
# ---------------------------------------------------------------------------
def test_spaces_and_first_step():
    env = AJLATTEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (4, env.obs_dim) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.single_observation_space.shape == (env.obs_dim,)
    assert env.action_space.shape == (4, 2) and env.single_action_space.shape == (2,)
    obs, reward, terminated, truncated, info = env.step(_forward())
    assert reward.shape == (4,) and terminated.shape == (4,) and terminated.dtype == bool
    assert isinstance(truncated, bool)
    np.testing.assert_allclose(info["agent_rewards"], reward)
    assert info["team_reward"] == pytest.approx(reward.sum())
    layout = env.observation_layout()
    assert max(s.stop for s in layout.values()) == env.obs_dim


def test_seeding_is_deterministic():
    def rollout(seed):
        env = AJLATTEnv(map_name="obstacles04")
        obs, _ = env.reset(seed=seed)
        out = [obs]
        rng = np.random.default_rng(1)
        for _ in range(15):
            obs, *_ = env.step(rng.uniform([0, -0.5], [0.8, 0.5], size=(4, 2)))
            out.append(obs)
        return np.stack(out)

    np.testing.assert_array_equal(rollout(3), rollout(3))
    assert not np.array_equal(rollout(3), rollout(4))


def test_config_seed_is_used_for_first_reset():
    a, _ = AJLATTEnv(seed=5).reset()
    b, _ = AJLATTEnv().reset(seed=5)
    np.testing.assert_array_equal(a, b)


def test_errors_and_action_handling():
    env = AJLATTEnv()
    with pytest.raises(RuntimeError):
        env.step(_forward())
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        env.step(np.zeros(8))
    with pytest.warns(UserWarning, match="extra rows"):
        env.step(np.zeros((6, 2)))


def test_torch_actions_are_accepted():
    torch = pytest.importorskip("torch")
    env = AJLATTEnv()
    env.reset(seed=0)
    env.step(0.1 * torch.ones(4, 2))


def test_clip_actions():
    env = AJLATTEnv(clip_actions=True, sigma_vR=0.0, sigma_wR=0.0)
    env.reset(seed=0)
    start = env.robot_true[0].state.copy()
    env.step(np.tile([10.0, 0.0], (4, 1)))
    travelled = np.linalg.norm(env.robot_true[0].state[:2] - start[:2])
    assert travelled == pytest.approx(env.config.max_linear_velocity * env.dt)


def test_truncation():
    env = AJLATTEnv(max_episode_steps=5)
    env.reset(seed=0)
    flags = [env.step(_forward(v=0.05))[3] for _ in range(5)]
    assert flags == [False] * 4 + [True]


def test_obstacle_collision_penalty():
    env = AJLATTEnv(map_name="obstacles04")
    env.reset(seed=0)
    grid = env.MAP
    cy, cx = np.argwhere(grid.map[20:-20, 20:-20] == 1)[0] + 20
    wall = grid.cell_to_se2([cx, cy])
    free_reward, free_collided = env.get_reward(env.RT_obs)
    env.robot_est[0].state = np.array([wall[0] - 0.1, wall[1], 0.0])
    reward, collided = env.get_reward(env.RT_obs)
    assert collided[0] and not free_collided[0]
    assert reward[0] == pytest.approx(free_reward[0] - env.config.obstacle_penalty)


def test_collision_flag_respects_config():
    for terminate in (True, False):
        env = AJLATTEnv(terminate_on_collision=terminate, sigma_vR=0.0, sigma_wR=0.0)
        env.reset(seed=0)
        grid = env.MAP
        cy, cx = np.argwhere(grid.map[20:-20, 20:-20] == 1)[0] + 20
        wall = grid.cell_to_se2([cx, cy])
        env.robot_est[0].state = np.array([wall[0] - 0.15, wall[1], 0.0])
        _, _, terminated, _, info = env.step(np.zeros((4, 2)))
        assert info["collisions"][0]
        assert terminated[0] == terminate


def test_coincident_estimates_stay_finite():
    env = AJLATTEnv()
    env.reset(seed=0)
    for est in env.robot_est:
        est.state = env.robot_est[0].state.copy()
    _, reward, _, _, info = env.step(np.zeros((4, 2)))
    assert np.all(np.isfinite(reward)) and not info["numerical_error"]


def test_dead_reckoning_covariance_grows():
    env = AJLATTEnv(use_update=False)
    env.reset(seed=0)
    traces = []
    for _ in range(10):
        _, _, _, _, info = env.step(_forward())
        traces.append(info["robot_cov_trace"])
    assert np.all(np.diff(np.array(traces), axis=0) > 0)


def test_episode_statistics_reset():
    env = AJLATTEnv()
    env.reset(seed=0)
    for _ in range(7):
        env.step(_forward())
    stats = env.episode_statistics()
    assert all(v.shape == (8, 4) for v in stats.values())
    env.reset(seed=1)
    assert all(v.shape == (1, 4) for v in env.episode_statistics().values())


@pytest.mark.parametrize("map_name", ["empty", "obstacles05", "obstacles02", "dynamic_map"])
def test_other_maps_run(map_name):
    env = AJLATTEnv(map_name=map_name)
    env.reset(seed=0)
    for _ in range(3):
        env.step(env.action_space.sample())


def test_dynamic_map_is_resampled_per_seed():
    env = AJLATTEnv(map_name="dynamic_map")
    env.reset(seed=1)
    first = env.MAP.map.copy()
    env.reset(seed=1)
    np.testing.assert_array_equal(first, env.MAP.map)
    env.reset(seed=2)
    assert not np.array_equal(first, env.MAP.map)


def test_registered_id_and_wrapper():
    env = env_lib.make("AJLATT-v0", map_name="obstacles05", num_robots=3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 7 + 4 * 2 + 5)
    wrapped = TeamRewardWrapper(env)
    _, reward, terminated, _, info = wrapped.step(np.zeros((3, 2)))
    assert isinstance(reward, float) and isinstance(terminated, bool)
    assert reward == pytest.approx(info["agent_rewards"].sum())
    assert info["agent_terminated"].shape == (3,)


def test_legacy_factory_and_callable_module():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = env_lib.ajlatt_env(map_name="obstacles04", num_Robot=3, render=False, some_ppo_flag=1)
    assert isinstance(env, AJLATTEnv) and env.num_robots == 3 and env.render_mode is None
    messages = " ".join(str(w.message) for w in caught)
    assert "num_robots" in messages and "some_ppo_flag" in messages
    assert isinstance(make(map_name="empty"), AJLATTEnv)
    assert "obstacles04" in env_lib.ajlatt_env.available_maps()


def test_target_policies_are_per_instance():
    a, b = AJLATTEnv(map_name="empty"), AJLATTEnv(map_name="empty")
    assert isinstance(a._target_policy, SinePolicy)
    a._target_policy.circling = True
    assert b._target_policy.circling is False
    a.reset(seed=0)
    assert a._target_policy.circling is False


# ---------------------------------------------------------------------------
# Estimation primitives
# ---------------------------------------------------------------------------
def test_se2_step_and_angle_wrapping():
    pose = se2_step(np.array([0.0, 0.0, np.pi - 0.05]), 1.0, (1.0, 0.1))
    assert -np.pi <= pose[2] < np.pi
    assert pose[0] == pytest.approx(np.cos(np.pi - 0.05))
    assert pi_to_pi(3 * np.pi) == pytest.approx(-np.pi)


def test_ekf_prediction_is_symmetric_psd():
    est = AgentEstimate(3, 0.5)
    est.reset([1.0, 2.0, 0.3], 0.1 * np.eye(3), np.random.default_rng(0))
    for _ in range(20):
        est.propagate((0.5, 0.2), 0.1, 0.05)
    np.testing.assert_allclose(est.cov, est.cov.T)
    assert np.linalg.eigvalsh(est.cov).min() > 0


def test_measurement_jacobians_match_finite_differences():
    xi, xj = np.array([1.0, 2.0, 0.4]), np.array([3.0, 1.0, -0.7])
    zhat, hi, hj = measurement_model(xi, xj)
    eps = 1e-6
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        num_i = (measurement_model(xi + d, xj)[0] - measurement_model(xi - d, xj)[0]) / (2 * eps)
        num_j = (measurement_model(xi, xj + d)[0] - measurement_model(xi, xj - d)[0]) / (2 * eps)
        np.testing.assert_allclose(hi[:, k], num_i, atol=1e-6)
        np.testing.assert_allclose(hj[:, k], num_j, atol=1e-6)
    assert zhat[0] == pytest.approx(np.hypot(2.0, -1.0))


def _random_psd(rng, d, rank=None):
    a = rng.normal(size=(d, rank or d))
    return a @ a.T


def test_ci_newton_is_optimal_and_on_simplex():
    rng = np.random.default_rng(0)
    for trial in range(60):
        d, n = int(rng.choice([2, 3])), int(rng.integers(2, 6))
        S = np.stack([_random_psd(rng, d) * 10 ** rng.uniform(-2, 2) for _ in range(n)])
        if trial % 3 == 0 and d == 3:
            S[:, 2, :] = S[:, :, 2] = 0.0  # unobservable third component
        c_newton = ci_weights(S, "newton")
        c_slsqp = ci_weights(S, "slsqp")
        assert c_newton.min() >= 0 and c_newton.sum() == pytest.approx(1.0)

        def objective(c, S=S):
            return np.trace(np.linalg.pinv(np.tensordot(c, S, axes=1)))

        assert objective(c_newton) <= objective(c_slsqp) * (1 + 1e-8)


def test_ci_degenerate_inputs():
    np.testing.assert_allclose(ci_weights(np.stack([np.eye(3)] * 4)), 0.25)
    np.testing.assert_array_equal(ci_weights(np.eye(2)[None]), [1.0])
    with pytest.raises(ValueError):
        ci_weights(np.eye(2)[None], solver="nope")
    S = np.stack([np.diag([4.0, 1.0]), np.diag([1.0, 4.0])])
    y = np.array([[4.0, 1.0], [1.0, 4.0]])
    cov, mean = covariance_intersection(S, y)
    np.testing.assert_allclose(cov, np.linalg.inv(np.tensordot([0.5, 0.5], S, axes=1)))
    np.testing.assert_allclose(mean, cov @ (y @ [0.5, 0.5]))
    info, vec = covariance_intersection(S, y, information_form=True)
    np.testing.assert_allclose(info, psd_inverse(cov))
    singular = np.diag([2.0, 0.0])
    np.testing.assert_allclose(psd_inverse(singular), np.diag([0.5, 0.0]))


def test_controllers():
    policy = WaypointPolicy([((-np.inf, np.inf), (10.0, 0.0))], velocity=0.3, k_theta=1.0)
    v, w = policy(np.array([0.0, 0.0, 0.5]))
    assert v == 0.3 and w < 0  # turn right towards the goal
    assert CirclePolicy()(np.zeros(3)) == (0.25, 0.15)
    assert isinstance(make_target_policy("auto", "obstacles04"), WaypointPolicy)
    assert isinstance(make_target_policy("auto", "unknown_map"), CirclePolicy)
    with pytest.raises(ValueError):
        make_target_policy("waypoints", "unknown_map")
    with pytest.raises(ValueError):
        make_target_policy("teleport")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_rgb_array_rendering(theme):
    set_theme(theme)
    try:
        env = AJLATTEnv(render_mode="rgb_array")
        env.reset(seed=0)
        frames = [env.render()]
        for _ in range(4):
            env.step(_forward(v=0.4, w=0.2))
            frames.append(env.render())
        frame = frames[-1]
        assert frame.shape == (620, 1100, 3) and frame.dtype == np.uint8
        assert frame.std() > 5 and not np.array_equal(frames[0], frames[-1])
        renderer = env._renderer
        for artist in renderer._dynamic_artists:
            artist.set_animated(False)
        full = figure_to_rgb(renderer.fig)
        assert np.abs(full.astype(int) - frame.astype(int)).mean() < 0.5
        env.close()
        env.close()
    finally:
        set_theme("dark")


def test_render_without_mode_returns_none():
    env = AJLATTEnv()
    env.reset(seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert env.render() is None


def test_gym_passive_checker_disabled_for_joint_api():
    spec = gym.registry["AJLATT-v0"]
    assert spec.disable_env_checker


def test_non_finite_actions_rejected():
    env = AJLATTEnv()
    env.reset(seed=0)
    bad = _forward()
    bad[1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        env.step(bad)


def test_builder_sample_is_named_after_file(tmp_path):
    from env_lib.ajlatt_env.maps import builder

    assert builder.main(["--create-sample", str(tmp_path / "my_map")]) == 0
    spec = tmp_path / "my_map.spec.yaml"
    assert "name: my_map" in spec.read_text()
    builder.main([str(spec)])
    assert (tmp_path / "my_map.yaml").exists() and (tmp_path / "my_map.cfg").exists()
    grid = env_lib.ajlatt_env.load_grid_map(tmp_path / "my_map")
    assert grid.map.shape == (181, 181)


# ---------------------------------------------------------------------------
# Exactness of the optimised numerical kernels
# ---------------------------------------------------------------------------
def _reference_weights_newton(S, max_iter=100):
    """Verbatim copy of the straightforward NumPy Newton solver (release 1.0)."""
    n = S.shape[0]

    def objective(c):
        fused = (c @ S.reshape(n, -1)).reshape(S.shape[1:])
        try:
            inverse = np.linalg.inv(fused)
        except np.linalg.LinAlgError:
            return np.inf, None
        value = np.trace(inverse)
        if not np.isfinite(value) or value <= 0:
            return np.inf, None
        return value, inverse

    c = np.full(n, 1.0 / n)
    f, p = objective(c)
    free = np.ones(n, dtype=bool)
    for _ in range(max_iter):
        b = S @ p
        a = p @ b
        grad = -np.einsum("nii->n", a)
        hess = 2.0 * a.reshape(n, -1) @ b.transpose(0, 2, 1).reshape(n, -1).T
        if free.all():
            idx, sub_hess, sub_grad = None, hess, grad
        else:
            idx = np.flatnonzero(free)
            sub_hess, sub_grad = hess[idx][:, idx], grad[idx]
        m = sub_grad.size
        kkt = np.empty((m + 1, m + 1))
        kkt[:m, :m] = sub_hess
        kkt[:m, m] = kkt[m, :m] = 1.0
        kkt[m, m] = 0.0
        rhs = np.append(-sub_grad, 0.0)
        try:
            solution = np.linalg.solve(kkt, rhs)
            if not np.all(np.isfinite(solution)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        if idx is None:
            step = solution[:m].copy()
        else:
            step = np.zeros(n)
            step[idx] = solution[:m]
        nu = solution[m]
        decrement = -float(grad @ step)
        accepted = False
        if decrement > 1e-14 * f and np.max(np.abs(step)) > 1e-13:
            decreasing = step < 0
            alpha_max = 1.0
            if decreasing.any():
                alpha_max = min(1.0, float(np.min(c[decreasing] / -step[decreasing])))
            alpha = alpha_max
            while alpha > 1e-12:
                trial = np.maximum(c + alpha * step, 0.0)
                trial /= trial.sum()
                f_trial, p_trial = objective(trial)
                if p_trial is not None and f_trial <= f - 1e-4 * alpha * decrement:
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                if alpha == alpha_max and alpha_max < 1.0:
                    blocked = decreasing & (c + alpha * step <= 1e-14)
                    free &= ~blocked
                    trial[blocked] = 0.0
                    trial /= trial.sum()
                    f_blocked, p_blocked = objective(trial)
                    if p_blocked is not None:
                        f_trial, p_trial = f_blocked, p_blocked
                    else:
                        free |= blocked
                c, f, p = trial, f_trial, p_trial
                continue
        multipliers = grad + nu
        fixed = np.flatnonzero(~free)
        tolerance = 1e-9 * max(1.0, float(np.max(np.abs(grad))))
        if fixed.size and multipliers[fixed].min() < -tolerance:
            free[fixed[np.argmin(multipliers[fixed])]] = True
            continue
        break
    return c


def test_ci_newton_is_bitwise_identical_to_reference():
    """The optimised solver performs the same floating-point operations.

    Nearly identical sources (neighbouring robots sharing estimates) make the
    KKT system ill-conditioned, so any rounding change would alter the
    weights; the optimised implementation must therefore match exactly.
    """
    rng = np.random.default_rng(3)
    for trial in range(150):
        d, n = int(rng.choice([2, 3])), int(rng.integers(2, 6))
        S = np.stack([_random_psd(rng, d) * 10 ** rng.uniform(-2, 2) for _ in range(n)])
        if trial % 4 == 0:
            S[1] = S[0] * (1.0 + 1e-10 * rng.normal())  # degenerate optimum
        if trial % 5 == 0:
            S[-1] = _random_psd(rng, d, rank=1)  # rank-deficient source
        np.testing.assert_array_equal(ci_weights(S), _reference_weights_newton(S))


def test_psd_inverse_matches_numpy():
    rng = np.random.default_rng(4)
    for _ in range(50):
        m = _random_psd(rng, 3) + 0.1 * np.eye(3)
        np.testing.assert_array_equal(psd_inverse(m), np.linalg.inv(m))
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(psd_inverse(singular), np.linalg.pinv(singular))
    np.testing.assert_allclose(psd_inverse([[2.0, 0.0], [0.0, 4.0]]), np.diag([0.5, 0.25]))


@pytest.mark.parametrize("map_name", ["obstacles04", "empty"])
def test_batched_ray_casting_matches_single_casts(map_name):
    grid = env_lib.ajlatt_env.load_grid_map(map_name, margin2wall=1.0)
    rng = np.random.default_rng(5)
    lo, hi = grid.mapmin - 1.0, grid.mapmax + 1.0
    poses = np.column_stack(
        [rng.uniform(lo[0], hi[0], 40), rng.uniform(lo[1], hi[1], 40), rng.uniform(-4, 4, 40)]
    )
    full, sensor = grid.closest_obstacles(poses, fov=[2 * np.pi, np.pi / 2], r_max=3.0)
    for pose, a, b in zip(poses, full, sensor):
        assert a == grid.get_closest_obstacle(pose, fov=2 * np.pi, r_max=3.0)
        assert b == grid.get_closest_obstacle(pose, fov=np.pi / 2, r_max=3.0)
    assert grid.closest_obstacles(poses[:3], fov=np.pi) == [
        grid.get_closest_obstacle(pose, fov=np.pi) for pose in poses[:3]
    ]
    assert grid.closest_obstacles(np.zeros((0, 3))) == []
