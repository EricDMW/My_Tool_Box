"""Tests for :mod:`env_lib.power_grid_env` (``PowerGrid-v0``)."""

from __future__ import annotations

import math
import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env

import env_lib
from env_lib.errors import ResetNeededError
from env_lib.power_grid_env import (
    DEFAULT_DROOP_GAIN,
    OBSERVATION_FEATURES,
    OBSERVATION_SCALE,
    TOPOLOGIES,
    PowerGridEnv,
    PowerGridVectorEnv,
    droop_policy,
)
from env_lib.utils import graphs, rendering
from env_lib.utils.rendering import figure_to_rgb

N = 16
QUIET = dict(initial_disturbances=0, disturbance_rate=0.0, noise_std=0.0)
INFO_KEYS = {
    "agent_rewards",
    "omega",
    "rocof",
    "max_abs_omega",
    "frequency_nadir",
    "peak_abs_omega",
    "coi_omega",
    "frequency_spread",
    "control_effort",
    "order_parameter",
    "total_disturbance",
    "tripped",
    "step",
    "time",
}


# ---------------------------------------------------------------------------
# Loop references written independently of the vectorised implementation
# ---------------------------------------------------------------------------
def reference_power_out(theta, weights):
    n = len(theta)
    return np.array(
        [sum(weights[i, j] * math.sin(theta[i] - theta[j]) for j in range(n)) for i in range(n)]
    )


def reference_step(theta, omega, injection, control, inertia, damping, weights, dt, substeps):
    """Classical RK4 of the swing equations, bus by bus."""

    def rhs(th, w):
        power = reference_power_out(th, weights)
        return w.copy(), (injection + control - damping * w - power) / inertia

    h = dt / substeps
    for _ in range(substeps):
        k1 = rhs(theta, omega)
        k2 = rhs(theta + 0.5 * h * k1[0], omega + 0.5 * h * k1[1])
        k3 = rhs(theta + 0.5 * h * k2[0], omega + 0.5 * h * k2[1])
        k4 = rhs(theta + h * k3[0], omega + h * k3[1])
        theta = theta + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        omega = omega + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return theta, omega


def newton_equilibrium(weights, injection, theta0, iterations=20):
    """Newton's method on the flow equations ``P = sum_j B_ij sin(theta_i - theta_j)``."""
    theta = theta0.copy()
    n = len(theta)
    for _ in range(iterations):
        mismatch = injection - reference_power_out(theta, weights)
        coupling = weights * np.cos(theta[:, None] - theta[None, :])
        jacobian = np.diag(coupling.sum(axis=1)) - coupling
        theta = theta + np.linalg.solve(jacobian + np.ones((n, n)) / n, mismatch)
    return theta - theta.mean()


def lyapunov_energy(env):
    """Kinetic plus potential energy of the swing equations around the equilibrium."""
    theta, star, weights = env.theta, env.equilibrium, env.susceptance
    kinetic = 0.5 * np.sum(env.inertia * env.omega**2)
    delta = theta[:, None] - theta[None, :]
    delta_star = star[:, None] - star[None, :]
    potential = -np.sum(env.nominal_injections * (theta - star)) + 0.5 * np.sum(
        weights * (np.cos(delta_star) - np.cos(delta))
    )
    return kinetic + potential


def rollout_vector(policy, num_envs=16, seed=0, **kwargs):
    """Mean team return of one episode per copy (no autoreset)."""
    envs = PowerGridVectorEnv(num_envs, autoreset_mode="disabled", **kwargs)
    obs, _ = envs.reset(seed=seed)
    returns = np.zeros(num_envs)
    done = np.zeros(num_envs, dtype=bool)
    tripped = np.zeros(num_envs, dtype=bool)
    rng = np.random.default_rng(seed)
    for _ in range(envs.max_steps):
        obs, rewards, terminated, truncated, _ = envs.step(policy(obs, rng, envs.u_max))
        returns[~done] += rewards[~done]
        tripped |= terminated & ~done
        done |= terminated | truncated
    return returns, tripped


@pytest.fixture
def light_theme():
    previous = rendering.get_theme()
    rendering.set_theme("light")
    try:
        yield
    finally:
        rendering.set_theme(previous)


# ---------------------------------------------------------------------------
# Spaces, observations and info
# ---------------------------------------------------------------------------
def test_default_spaces_and_dtypes():
    env = PowerGridEnv()
    assert env.n_buses == env.n_agents == N
    assert env.observation_space == Box(-np.inf, np.inf, (N, 10), np.float32)
    assert env.action_space == Box(-0.1, 0.1, (N,), np.float32)
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == (N, 10) and obs.flags["C_CONTIGUOUS"]
    assert env.observation_space.contains(obs)
    assert set(info) == INFO_KEYS
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs) and np.all(np.isfinite(obs))
    assert type(reward) is float
    assert type(terminated) is bool and type(truncated) is bool
    assert info["agent_rewards"].shape == (N,) and info["agent_rewards"].dtype == np.float64
    assert info["omega"].shape == info["rocof"].shape == (N,)
    for key in INFO_KEYS - {"agent_rewards", "omega", "rocof", "tripped", "step"}:
        assert isinstance(info[key], float), key
    assert type(info["tripped"]) is bool and type(info["step"]) is int
    assert reward == pytest.approx(info["agent_rewards"].sum())


def test_observation_layout_and_values():
    env = PowerGridEnv(u_max=np.linspace(0.05, 0.2, N), neighbourhood=2)
    layout = env.observation_layout
    assert list(layout) == list(OBSERVATION_FEATURES)
    assert [layout[k] for k in OBSERVATION_FEATURES] == [slice(i, i + 1) for i in range(10)]
    assert env.observation_scale == OBSERVATION_SCALE
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(3):
        before = env.omega
        obs, *_, info = env.step(rng.uniform(-0.3, 0.3, N))
    col = {k: obs[:, layout[k]][:, 0].astype(np.float64) / OBSERVATION_SCALE[k] for k in layout}
    np.testing.assert_allclose(col["omega"], env.omega, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(col["rocof"], (env.omega - before) / env.dt, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(col["disturbance"], env.disturbance, rtol=1e-6, atol=1e-7)
    power = reference_power_out(env.theta, env.susceptance) - env.nominal_injections
    np.testing.assert_allclose(col["power_flow"], power, rtol=1e-5, atol=1e-7)
    applied = np.clip(col["action"], -env.u_max, env.u_max)
    np.testing.assert_allclose(col["action"], applied, atol=1e-7)
    np.testing.assert_allclose(col["capacity"], env.u_max, rtol=1e-6)
    np.testing.assert_allclose(col["inertia"], env.inertia_constants, rtol=1e-6)
    # Aggregates over the 2-hop neighbourhood, bus by bus.
    reach = graphs.k_hop_adjacency(env.adjacency, 2)
    for i in range(N):
        hood = np.flatnonzero(reach[i])
        assert col["neighbour_omega_mean"][i] == pytest.approx(env.omega[hood].mean(), abs=1e-6)
        assert col["neighbour_omega_max"][i] == pytest.approx(
            np.abs(env.omega[hood]).max(), abs=1e-6
        )
        assert col["neighbour_action_mean"][i] == pytest.approx(
            col["action"][hood].mean(), abs=1e-6
        )


def test_first_observation_and_info():
    env = PowerGridEnv(initial_disturbances=3)
    obs, info = env.reset(seed=2)
    layout = env.observation_layout
    assert np.count_nonzero(env.disturbance) == 3
    np.testing.assert_array_equal(obs[:, layout["omega"]], 0.0)
    np.testing.assert_array_equal(obs[:, layout["rocof"]], 0.0)
    np.testing.assert_array_equal(obs[:, layout["action"]], 0.0)
    np.testing.assert_allclose(obs[:, layout["power_flow"]], 0.0, atol=1e-6)
    assert info["step"] == 0 and info["time"] == 0.0 and not info["tripped"]
    assert info["total_disturbance"] == pytest.approx(env.disturbance.sum())
    np.testing.assert_array_equal(info["agent_rewards"], np.zeros(N))


def test_check_env():
    env = PowerGridEnv(noise_std=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The observation space is deliberately unbounded (physical measurements).
        warnings.filterwarnings("ignore", message=r".*Box observation space m\w+ value is")
        check_env(env, skip_render_check=True)


# ---------------------------------------------------------------------------
# Seeding and network
# ---------------------------------------------------------------------------
def test_seeding_determinism():
    kwargs = dict(noise_std=0.02, disturbance_rate=2.0)
    env_a, env_b = PowerGridEnv(**kwargs), PowerGridEnv(**kwargs)
    obs_a, _ = env_a.reset(seed=7)
    obs_b, _ = env_b.reset(seed=7)
    np.testing.assert_array_equal(obs_a, obs_b)
    rng = np.random.default_rng(3)
    for _ in range(40):
        action = rng.uniform(-0.1, 0.1, N)
        out_a, out_b = env_a.step(action), env_b.step(action)
        np.testing.assert_array_equal(out_a[0], out_b[0])
        assert out_a[1] == out_b[1]
    obs_c, _ = env_a.reset(seed=8)
    assert not np.array_equal(env_a.disturbance, env_b.disturbance) or not np.array_equal(
        obs_a, obs_c
    )


def test_benchmark_network_is_fixed_by_network_seed():
    a, b = PowerGridEnv(), PowerGridEnv()
    np.testing.assert_array_equal(a.susceptance, b.susceptance)
    np.testing.assert_array_equal(a.inertia, b.inertia)
    a.reset(seed=0)
    first = a.susceptance
    a.reset(seed=1)
    np.testing.assert_array_equal(a.susceptance, first)
    c = PowerGridEnv(network_seed=1)
    assert not np.array_equal(c.susceptance, a.susceptance)
    # Physical parameters within their documented ranges.
    omega0 = 2 * math.pi * 50.0
    assert np.all((a.inertia_constants >= 2.0) & (a.inertia_constants <= 8.0))
    np.testing.assert_allclose(a.inertia, 2 * a.inertia_constants / omega0)
    assert np.all((a.damping * omega0 >= 2.0) & (a.damping * omega0 <= 6.0))
    weights = a.susceptance[a.adjacency]
    assert np.all((weights >= 0.3) & (weights <= 0.8))
    np.testing.assert_array_equal(a.susceptance > 0, a.adjacency)
    assert graphs.is_connected(a.adjacency)


def test_equilibrium_is_exact_and_matches_newton():
    env = PowerGridEnv(line_loading=0.6)
    star, weights, injection = env.equilibrium, env.susceptance, env.nominal_injections
    assert injection.sum() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(reference_power_out(star, weights), injection, atol=1e-12)
    edges = env.edges
    loading = np.abs(np.sin(star[edges[:, 0]] - star[edges[:, 1]]))
    assert loading.max() == pytest.approx(0.6, abs=1e-12)
    # Newton from the linearised (DC) solution converges to the same angles.
    laplacian = graphs.laplacian(weights)
    start = np.linalg.lstsq(laplacian, injection, rcond=None)[0]
    np.testing.assert_allclose(newton_equilibrium(weights, injection, start), star, atol=1e-10)


def test_equilibrium_start_is_stationary_without_disturbance():
    env = PowerGridEnv(**QUIET)
    obs, info = env.reset(seed=0)
    assert info["order_parameter"] < 1.0
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(np.zeros(N))
    assert truncated and not terminated
    assert np.abs(env.omega).max() < 1e-12
    np.testing.assert_allclose(env.theta, env.equilibrium, atol=1e-12)
    assert abs(reward) < 1e-20


def test_damping_dissipates_energy():
    env = PowerGridEnv(**QUIET)
    env.reset(seed=0, options={"omega": np.linspace(-0.5, 0.5, N)})
    energy = [lyapunov_energy(env)]
    for _ in range(150):
        env.step(np.zeros(N))
        energy.append(lyapunov_energy(env))
    assert np.all(np.diff(energy) < 1e-12)
    assert energy[-1] < 0.2 * energy[0]
    # With zero damping the energy is conserved (up to the RK4 error).
    env = PowerGridEnv(damping_range=(0.0, 0.0), substeps=8, **QUIET)
    env.reset(seed=0, options={"omega": np.linspace(-0.5, 0.5, N)})
    start = lyapunov_energy(env)
    for _ in range(40):
        env.step(np.zeros(N))
    assert lyapunov_energy(env) == pytest.approx(start, rel=1e-6)


def test_dynamics_match_loop_reference():
    env = PowerGridEnv(n_buses=7, topology="erdos_renyi", substeps=3, noise_std=0.01)
    env.reset(seed=4, options={"omega": np.linspace(-0.3, 0.3, 7)})
    rng = np.random.default_rng(1)
    for _ in range(5):
        theta, omega, drive = env.theta, env.omega, env.nominal_injections + env.disturbance
        action = rng.uniform(-0.2, 0.2, 7)
        control = np.clip(action.astype(np.float32).astype(np.float64), -0.1, 0.1)
        expected = reference_step(
            theta, omega, drive, control, env.inertia, env.damping, env.susceptance, env.dt, 3
        )
        _, reward, _, _, info = env.step(action)
        np.testing.assert_allclose(env.theta, expected[0], atol=1e-12)
        np.testing.assert_allclose(env.omega, expected[1], atol=1e-12)
        rocof = (expected[1] - omega) / env.dt
        agent = -(expected[1] ** 2 + 0.1 * rocof**2 + 10.0 * control**2) * env.dt
        np.testing.assert_allclose(info["agent_rewards"], agent, rtol=1e-9, atol=1e-15)
        assert info["control_effort"] == pytest.approx(np.abs(control).sum())
        coi = np.sum(env.inertia * env.omega) / env.inertia.sum()
        assert info["coi_omega"] == pytest.approx(coi)
        order = abs(np.exp(1j * env.theta).mean())
        assert info["order_parameter"] == pytest.approx(order)


def test_rk4_converges_with_substeps():
    def final_omega(substeps):
        env = PowerGridEnv(substeps=substeps, disturbance_rate=0.0)
        env.reset(seed=1)
        for _ in range(60):
            env.step(np.zeros(N))
        return env.omega

    reference = final_omega(16)
    error_1 = np.abs(final_omega(1) - reference).max()
    error_2 = np.abs(final_omega(2) - reference).max()
    assert error_2 < error_1 / 8  # fourth order: halving h divides the error by about 16
    assert error_2 < 1e-3 * np.abs(reference).max()


def test_randomize_network():
    env = PowerGridEnv(randomize_network=True, topology="erdos_renyi")
    env.reset(seed=0)
    first = (env.adjacency, env.susceptance, env.inertia)
    env.reset(seed=1)
    second = (env.adjacency, env.susceptance, env.inertia)
    assert not np.array_equal(first[1], second[1])
    assert not np.array_equal(first[0], second[0])
    assert not np.array_equal(first[2], second[2])
    env.reset(seed=0)
    np.testing.assert_array_equal(env.susceptance, first[1])  # reproducible from the seed
    # Every copy of the vector env owns a network; the equilibrium stays exact.
    envs = PowerGridVectorEnv(4, randomize_network=True, **QUIET)
    envs.reset(seed=3)
    adjacency, inertia = envs.adjacency, envs.inertia
    assert len({adjacency[b].tobytes() for b in range(4)}) == 4
    assert len({inertia[b].tobytes() for b in range(4)}) == 4
    for _ in range(20):
        _, _, _, _, infos = envs.step(np.zeros((4, N), np.float32))
    assert np.abs(infos["omega"]).max() < 1e-12


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_all_topologies(topology):
    env = PowerGridEnv(n_buses=12, topology=topology, network_seed=3)
    obs, _ = env.reset(seed=0)
    assert graphs.is_connected(env.adjacency)
    for _ in range(5):
        obs, *_ = env.step(droop_policy(obs))
    assert env.observation_space.contains(obs)
    assert env.positions.shape == (12, 2) and np.abs(env.positions).max() <= 1.0 + 1e-12


def test_scales_to_hundreds_of_buses():
    env = PowerGridEnv(n_buses=300)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, reward, terminated, _, _ = env.step(droop_policy(obs))
    assert obs.shape == (300, 10) and np.isfinite(reward) and not terminated


# ---------------------------------------------------------------------------
# Disturbances, termination and truncation
# ---------------------------------------------------------------------------
def test_disturbance_statistics():
    envs = PowerGridVectorEnv(
        256,
        initial_disturbances=2,
        disturbance_rate=4.0,
        disturbance_magnitude=(0.1, 0.2),
        load_increase_prob=1.0,
        autoreset_mode="disabled",
    )
    envs.reset(seed=0)
    initial = envs.disturbance
    assert np.all(np.count_nonzero(initial, axis=1) == 2)
    nonzero = initial[initial != 0]
    assert np.all((nonzero <= -0.1) & (nonzero >= -0.2))  # load increases only
    for _ in range(20):
        envs.step(np.zeros((256, N), np.float32))
    events = np.count_nonzero(np.diff(np.stack([initial, envs.disturbance]), axis=0))
    assert events > 0
    total_events = -(envs.disturbance.sum(axis=1) - initial.sum(axis=1)) / 0.15
    expected = 20 * -math.expm1(-4.0 * 0.05)  # at most one event per step
    assert total_events.mean() == pytest.approx(expected, rel=0.1)


def test_ornstein_uhlenbeck_fluctuations():
    envs = PowerGridVectorEnv(
        512, initial_disturbances=0, disturbance_rate=0.0, noise_std=0.02, noise_tau=0.5
    )
    envs.reset(seed=0)
    first = envs.disturbance
    assert first.std() == pytest.approx(0.02, rel=0.05)
    envs.step(np.zeros((512, N), np.float32))
    second = envs.disturbance
    correlation = np.corrcoef(first.ravel(), second.ravel())[0, 1]
    assert correlation == pytest.approx(math.exp(-0.05 / 0.5), abs=0.02)
    assert second.std() == pytest.approx(0.02, rel=0.05)


def test_trip_terminates_with_penalty():
    env = PowerGridEnv(**QUIET)
    limit = env.omega_limit
    assert limit == pytest.approx(2 * math.pi * 0.5)
    omega = np.zeros(N)
    omega[3] = 1.2 * limit
    env.reset(seed=0, options={"omega": omega})
    _, reward, terminated, truncated, info = env.step(np.zeros(N))
    assert terminated and not truncated and info["tripped"]
    assert info["max_abs_omega"] > limit
    assert np.all(info["agent_rewards"] < -100.0)
    physical = -(info["omega"] ** 2 + 0.1 * info["rocof"] ** 2) * env.dt
    np.testing.assert_allclose(info["agent_rewards"], physical - 100.0)
    assert reward == pytest.approx(info["agent_rewards"].sum())


def test_truncation_and_nadir_tracking():
    env = PowerGridEnv(max_steps=30, initial_disturbances=1, load_increase_prob=1.0)
    env.reset(seed=0)
    lowest, peak = 0.0, 0.0
    for step in range(1, 31):
        _, _, terminated, truncated, info = env.step(np.zeros(N))
        lowest, peak = min(lowest, env.omega.min()), max(peak, np.abs(env.omega).max())
        assert not terminated and truncated == (step == 30)
        assert info["step"] == step and info["time"] == pytest.approx(step * 0.05)
        assert info["frequency_nadir"] == pytest.approx(lowest)
        assert info["peak_abs_omega"] == pytest.approx(peak)
    assert info["frequency_nadir"] < 0.0


# ---------------------------------------------------------------------------
# Baseline controller
# ---------------------------------------------------------------------------
def test_droop_policy_shapes_and_clipping():
    env = PowerGridEnv(u_max=0.05)
    obs, _ = env.reset(seed=0)
    obs, *_ = env.step(np.zeros(N))
    action = droop_policy(obs)
    assert action.shape == (N,) and action.dtype == np.float32
    omega = obs[:, 0].astype(np.float64)
    np.testing.assert_allclose(
        action, np.clip(-DEFAULT_DROOP_GAIN * omega, -0.05, 0.05), rtol=1e-6, atol=1e-8
    )
    batch = np.stack([obs, obs, obs])[None]  # (1, 3, n, obs_dim)
    batch[0, 1, :, 0] *= 3.0
    batch[0, 2, :, 0] *= -1.0
    batched = droop_policy(batch, gain=50.0)
    assert batched.shape == (1, 3, N)
    assert np.all(np.abs(batched) <= 0.05 + 1e-7)
    np.testing.assert_allclose(batched[0, 0], droop_policy(obs, gain=50.0))
    np.testing.assert_array_equal(droop_policy(obs, gain=0.0), np.zeros(N, np.float32))
    with pytest.raises(ValueError):
        droop_policy(obs[:, :5])
    with pytest.raises(ValueError):
        droop_policy(obs[0])
    with pytest.raises(ValueError):
        droop_policy(obs, gain=-1.0)


def test_droop_beats_zero_and_random_control():
    def zero(obs, rng, u_max):
        return np.zeros(obs.shape[:-1], np.float32)

    def random(obs, rng, u_max):
        return rng.uniform(-u_max, u_max, obs.shape[:-1]).astype(np.float32)

    def droop(obs, rng, u_max):
        return droop_policy(obs)

    results = {name: rollout_vector(p) for name, p in [("zero", zero), ("random", random),
                                                        ("droop", droop)]}  # fmt: skip
    means = {name: returns.mean() for name, (returns, _) in results.items()}
    assert means["droop"] > 10.0 * means["zero"]  # returns are negative
    assert means["zero"] > means["random"]
    droop_returns, droop_trips = results["droop"]
    assert not droop_trips.any()
    assert np.all(droop_returns > results["zero"][0])


# ---------------------------------------------------------------------------
# Vector environment
# ---------------------------------------------------------------------------
def test_vector_env_shapes_and_infos():
    envs = PowerGridVectorEnv(5, n_buses=10)
    assert envs.observation_space.shape == (5, 10, 10)
    assert envs.action_space.shape == (5, 10)
    obs, infos = envs.reset(seed=0)
    assert obs.shape == (5, 10, 10) and obs.dtype == np.float32
    assert set(infos) >= INFO_KEYS and infos["_omega"].all()
    obs, rewards, terminated, truncated, infos = envs.step(envs.action_space.sample())
    assert envs.observation_space.contains(obs)
    assert rewards.shape == (5,) and rewards.dtype == np.float64
    assert terminated.dtype == bool and truncated.dtype == bool
    assert infos["agent_rewards"].shape == (5, 10)
    np.testing.assert_allclose(rewards, infos["agent_rewards"].sum(axis=1))
    assert infos["omega"].shape == (5, 10) and infos["max_abs_omega"].shape == (5,)
    assert envs.theta.shape == envs.omega.shape == envs.disturbance.shape == (5, 10)
    assert envs.adjacency.shape == (5, 10, 10) and envs.equilibrium.shape == (5, 10)


def test_vector_copy_zero_matches_single_env():
    kwargs = dict(disturbance_rate=0.0, noise_std=0.0, n_buses=12)
    single, vector = PowerGridEnv(**kwargs), PowerGridVectorEnv(6, **kwargs)
    rng = np.random.default_rng(0)
    options = {
        "theta": single.equilibrium + 0.05 * rng.standard_normal(12),
        "omega": 0.2 * rng.standard_normal(12),
        "disturbance": 0.1 * rng.standard_normal(12),
    }
    obs_single, info_single = single.reset(seed=0, options=options)
    obs_vector, info_vector = vector.reset(seed=123, options=options)
    np.testing.assert_array_equal(obs_single, obs_vector[0])
    np.testing.assert_array_equal(obs_vector[0], obs_vector[5])
    for _ in range(60):
        actions = rng.uniform(-0.08, 0.08, (6, 12)).astype(np.float32)
        obs_single, reward, terminated, truncated, info = single.step(actions[0])
        obs_vector, rewards, terminations, truncations, infos = vector.step(actions)
        np.testing.assert_allclose(obs_single, obs_vector[0], rtol=1e-6, atol=1e-7)
        assert reward == pytest.approx(rewards[0], rel=1e-9, abs=1e-12)
        assert terminated == terminations[0] and truncated == truncations[0]
        for key in ("omega", "agent_rewards"):
            np.testing.assert_allclose(info[key], infos[key][0], rtol=1e-9, atol=1e-12)
        assert not terminated


def test_vector_reset_options_per_copy():
    envs = PowerGridVectorEnv(3, **QUIET)
    omega = np.zeros((3, N))
    omega[1, 0] = 0.3
    envs.reset(seed=0, options={"omega": omega})
    np.testing.assert_array_equal(envs.omega, omega)
    with pytest.raises(ValueError):
        envs.reset(options={"omega": np.zeros((2, N))})


def test_vector_next_step_autoreset():
    envs = PowerGridVectorEnv(3, max_steps=50, **QUIET)
    omega = np.zeros((3, N))
    omega[1, 2] = 4.0  # above the trip limit of pi rad/s
    envs.reset(seed=0, options={"omega": omega})
    zero = np.zeros((3, N), np.float32)
    _, rewards, terminated, truncated, infos = envs.step(zero)
    assert terminated.tolist() == [False, True, False]
    assert infos["tripped"].tolist() == [False, True, False]
    assert rewards[1] < -100.0 * N
    obs, rewards, terminated, _, infos = envs.step(zero)
    # Copy 1 restarted from the equilibrium: zero reward, step counter 0.
    assert rewards[1] == 0.0 and not terminated[1]
    assert infos["step"].tolist() == [2, 0, 2]
    np.testing.assert_array_equal(obs[1, :, 0], 0.0)
    for _ in range(48):
        _, _, terminated, truncated, infos = envs.step(zero)
    assert truncated.tolist() == [True, False, True]


def test_vector_same_step_autoreset():
    envs = PowerGridVectorEnv(2, autoreset_mode="same_step", **QUIET)
    omega = np.zeros((2, N))
    omega[0, 0] = 4.0
    envs.reset(seed=0, options={"omega": omega})
    obs, _, terminated, _, infos = envs.step(np.zeros((2, N), np.float32))
    assert terminated.tolist() == [True, False]
    assert infos["_final_obs"].tolist() == [True, False]
    assert np.abs(infos["final_obs"][0][:, 0]).max() > np.pi
    np.testing.assert_array_equal(obs[0, :, 0], 0.0)
    final_info = infos["final_info"]  # batched info dict masked by "_final_info"
    assert final_info["tripped"][0] and final_info["step"][0] == 1
    # The returned info of copy 0 already describes its new episode.
    assert infos["step"].tolist() == [0, 1] and not infos["tripped"][0]


def test_make_and_make_vec_use_native_classes():
    env = env_lib.make("PowerGrid-v0")
    assert isinstance(env.unwrapped, PowerGridEnv)
    assert env_lib.PowerGridEnv is PowerGridEnv
    assert env_lib.PowerGridVectorEnv is PowerGridVectorEnv
    obs, _ = env.reset(seed=0)
    obs, *_ = env.step(droop_policy(obs))
    assert env.observation_space.contains(obs)
    env.close()
    envs = env_lib.make_vec("PowerGrid-v0", 8)
    assert isinstance(envs, PowerGridVectorEnv) and envs.num_envs == 8
    obs, _ = envs.reset(seed=0)
    obs, *_ = envs.step(droop_policy(obs))
    assert obs.shape == (8, N, 10)
    envs = env_lib.make_vec("PowerGrid-v0", 2, n_buses=6, autoreset_mode="same_step")
    assert envs.autoreset_mode == "same_step" and envs.single_action_space.shape == (6,)
    sync = env_lib.make_vec("PowerGrid-v0", 2, vectorization_mode="sync")
    assert isinstance(sync, gym.vector.SyncVectorEnv)
    sync.close()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_buses": 1},
        {"topology": "tree"},
        {"topology_kwargs": {"degree": 3}},
        {"topology": "small_world", "n_buses": 4},
        {"network_seed": -1},
        {"inertia_range": (0.0, 1.0)},
        {"inertia_range": (3.0, 2.0)},
        {"damping_range": (-1.0, 1.0)},
        {"susceptance_range": (0.0, 1.0)},
        {"line_loading": 1.0},
        {"line_loading": -0.1},
        {"nominal_frequency": 0.0},
        {"u_max": 0.0},
        {"u_max": [0.1, 0.2]},
        {"u_max": float("nan")},
        {"dt": 0.0},
        {"substeps": 0},
        {"max_steps": 0},
        {"initial_disturbances": 17},
        {"disturbance_rate": -1.0},
        {"disturbance_magnitude": (0.2, 0.1)},
        {"load_increase_prob": 1.5},
        {"noise_std": -0.1},
        {"noise_tau": 0.0},
        {"frequency_limit": 0.0},
        {"w_freq": -1.0},
        {"w_u": float("inf")},
        {"trip_penalty": -1.0},
        {"neighbourhood": 0},
        {"render_mode": "ansi"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        PowerGridEnv(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_buses": 16.0},
        {"substeps": "2"},
        {"dt": None},
        {"randomize_network": 1},
        {"inertia_range": 2.0},
        {"unknown_argument": 1},
    ],
)
def test_wrong_argument_types_raise_type_error(kwargs):
    with pytest.raises(TypeError):
        PowerGridEnv(**kwargs)
    with pytest.raises(TypeError):
        PowerGridVectorEnv(2, **kwargs)


def test_large_rk4_step_warns():
    with pytest.warns(RuntimeWarning, match="substeps"):
        PowerGridEnv(substeps=1, dt=0.25, susceptance_range=(2.0, 3.0))


def test_reset_needed_and_action_validation():
    env = PowerGridEnv(render_mode="rgb_array")
    with pytest.raises(ResetNeededError):
        env.step(np.zeros(N))
    with pytest.raises(ResetNeededError):
        _ = env.omega
    with pytest.raises(ResetNeededError):
        env.render()
    envs = PowerGridVectorEnv(2)
    with pytest.raises(ResetNeededError):
        envs.step(np.zeros((2, N)))
    with pytest.raises(ResetNeededError):
        _ = envs.theta
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.full(N, np.nan))
    with pytest.raises(ValueError):
        env.step(np.array([np.inf] + [0.0] * (N - 1)))
    with pytest.raises(ValueError):
        env.step(np.zeros(N + 1))
    env.step(np.zeros((N, 1)))  # any array with n entries is accepted
    envs.reset(seed=0)
    with pytest.raises(ValueError):
        envs.step(np.full((2, N), np.nan))
    with pytest.raises(ValueError):
        env.reset(options={"omega": np.zeros(3)})
    with pytest.raises(ValueError):
        env.reset(options={"theta": np.full(N, np.nan)})
    env.close()


def test_unknown_reset_options_are_ignored_with_a_warning():
    # Generic tools pass their own keys, e.g. PettingZoo's parallel_api_test
    # resets with options={"options": 1}.
    env = PowerGridEnv(**QUIET)
    reference, _ = env.reset(seed=0, options={"omega": np.full(N, 0.1)})
    with pytest.warns(UserWarning, match="ignoring unknown reset option") as record:
        obs, info = env.reset(seed=0, options={"options": 1, "omega": np.full(N, 0.1)})
    assert record[0].filename == __file__  # the warning points at the caller
    np.testing.assert_array_equal(obs, reference)
    assert info["step"] == 0
    with pytest.warns(UserWarning, match="ignoring unknown reset option"):
        env.reset(options={"options": 1})
    np.testing.assert_allclose(env.omega, 0.0)
    with pytest.warns(UserWarning), pytest.raises(ValueError):
        env.reset(options={"options": 1, "omega": np.zeros(3)})  # known keys stay strict
    envs = PowerGridVectorEnv(3, **QUIET)
    with pytest.warns(UserWarning, match="ignoring unknown reset option") as record:
        obs, _ = envs.reset(seed=0, options={"options": 1})
    assert record[0].filename == __file__
    assert obs.shape == (3, N, 10)
    envs.step(np.zeros((3, N), np.float32))
    with pytest.warns(UserWarning, match="ignoring unknown reset option"):
        envs.reset(options={"reset_mask": np.array([True, False, False]), "seed": 3})
    assert envs._core.steps.tolist() == [0, 1, 1]


def test_actions_are_clipped_to_heterogeneous_bounds():
    u_max = np.linspace(0.02, 0.2, N)
    env = PowerGridEnv(u_max=u_max, **QUIET)
    np.testing.assert_allclose(env.action_space.high, u_max.astype(np.float32))
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.ones(N))
    assert info["control_effort"] == pytest.approx(u_max.sum())
    obs, *_ = env.step(-np.ones(N))
    np.testing.assert_allclose(obs[:, 4] / 10.0, -u_max, rtol=1e-6)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_episode(env, steps=5, seed=16):
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    for _ in range(steps):
        obs, *_ = env.step(droop_policy(obs))
        frames.append(env.render())
    return frames


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_rgb_array_frames(theme):
    previous = rendering.get_theme()
    rendering.set_theme(theme)
    try:
        env = PowerGridEnv(render_mode="rgb_array")
        frames = _render_episode(env)
        env.close()
    finally:
        rendering.set_theme(previous)
    for frame in frames:
        assert frame.shape == (560, 1000, 3) and frame.dtype == np.uint8
        assert frame.std() > 5.0
    assert not np.array_equal(frames[1], frames[-1])
    expected = [int(rendering.get_theme(theme).background[k : k + 2], 16) for k in (1, 3, 5)]
    np.testing.assert_allclose(frames[0][2, 2], expected, atol=2)


def test_blitted_frame_matches_full_redraw(light_theme):
    env = PowerGridEnv(render_mode="rgb_array")
    frames = _render_episode(env, steps=8)
    renderer = env._renderer
    for artist in renderer._dynamic_artists:
        artist.set_animated(False)
    full = figure_to_rgb(renderer.fig)
    assert np.abs(frames[-1].astype(int) - full.astype(int)).mean() < 0.5
    env.close()


def test_render_after_reset_clears_history():
    env = PowerGridEnv(render_mode="rgb_array")
    _render_episode(env, steps=4)
    env.reset(seed=1)
    env.render()
    assert np.count_nonzero(~np.isnan(env._renderer._coi_hist)) == 1
    env.close()


def test_vector_env_renders_copy_zero():
    envs = PowerGridVectorEnv(3, n_buses=8, render_mode="rgb_array")
    envs.reset(seed=0)
    envs.step(np.zeros((3, 8), np.float32))
    frame = envs.render()
    assert frame.shape == (560, 1000, 3)
    envs.close()
    assert PowerGridVectorEnv(2).render() is None


def test_render_without_mode_warns_once_and_close_is_idempotent():
    env = PowerGridEnv()
    env.reset(seed=0)
    with pytest.warns(UserWarning, match="render_mode"):
        assert env.render() is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert env.render() is None
    env = PowerGridEnv(render_mode="rgb_array")
    env.close()
    env.reset(seed=0)
    env.render()
    env.close()
    env.close()
    assert env.render().shape == (560, 1000, 3)  # re-creates the figure
    env.close()


def test_human_mode_runs_headless():
    env = PowerGridEnv(render_mode="human", max_steps=2, n_buses=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # non-interactive Agg backend
        obs, _ = env.reset(seed=0)
        env.step(droop_policy(obs))
        assert env.render() is None
    env.close()
