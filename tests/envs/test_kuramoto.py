"""Tests for the Kuramoto oscillator environments (NumPy and PyTorch backends)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

import env_lib
from env_lib.kos_env import KuramotoOscillatorEnv
from env_lib.kos_env._common import build_topology, topology_edges
from env_lib.utils import rendering

KURAMOTO_IDS = [env_id for env_id in env_lib.list_envs() if env_id.startswith("Kuramoto")]
INFO_KEYS = {
    "order_parameter",
    "phase_coherence",
    "natural_frequencies",
    "phases",
    "coupling_matrix",
    "dphases_dt",
    "step_count",
    "device",
    "n_agents",
    "agent_rewards",
}


def torch_env_class():
    pytest.importorskip("torch")
    from env_lib.kos_env import KuramotoOscillatorEnvTorch

    return KuramotoOscillatorEnvTorch


def naive_dynamics(phases, natural_frequencies, coupling, control, normalize=False):
    """Reference implementation: explicit double loop over the full coupling matrix."""
    n = len(phases)
    out = np.empty(n)
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += coupling[i, j] * np.sin(phases[j] - phases[i])
        out[i] = natural_frequencies[i] + control[i] + (total / n if normalize else total)
    return out


def angle_diff(a, b):
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def synced_state(n):
    return {"phases": np.full(n, 0.3), "natural_frequencies": np.full(n, 1.0)}


def spread_state(n):
    return {
        "phases": np.linspace(-np.pi, np.pi, n, endpoint=False),
        "natural_frequencies": np.linspace(0.5, 2.0, n),
    }


# ---------------------------------------------------------------------------
# Spaces, observations and info
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("coupling_mode", ["dynamic", "constant"])
def test_spaces_and_observation_layout(coupling_mode):
    n = 6
    env = KuramotoOscillatorEnv(n_oscillators=n, coupling_mode=coupling_mode)
    m = n * (n - 1) // 2 if coupling_mode == "dynamic" else 0
    assert env.observation_space.shape == (3 * n + m,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (n + m,)
    assert env.action_space.dtype == np.float32
    assert env.n_couplings == n * (n - 1) // 2

    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert type(terminated) is bool and type(truncated) is bool

    np.testing.assert_allclose(obs[:n], env.phases.astype(np.float32))
    np.testing.assert_allclose(obs[n : 2 * n], env.natural_frequencies.astype(np.float32))
    np.testing.assert_allclose(obs[-n:], np.clip(action[:n], -1, 1), atol=1e-7)
    if coupling_mode == "dynamic":
        np.testing.assert_allclose(obs[2 * n : 2 * n + m], np.clip(action[n:], 0, 5), atol=1e-6)
    assert np.all(env.phases >= -np.pi) and np.all(env.phases < np.pi)


def test_info_keys_and_agent_rewards():
    env = KuramotoOscillatorEnv(n_oscillators=5, n_agents=3)
    _, info = env.reset(seed=1)
    assert {"order_parameter", "phase_coherence", "phases", "natural_frequencies"} <= set(info)
    _, reward, _, _, info = env.step(env.action_space.sample())
    assert INFO_KEYS <= set(info)
    assert info["agent_rewards"].shape == (3,) and info["agent_rewards"].dtype == np.float64
    np.testing.assert_allclose(info["agent_rewards"], reward)
    assert info["coupling_matrix"].shape == (5, 5)
    assert info["dphases_dt"].shape == (5,)
    assert info["step_count"] == 1 and info["n_agents"] == 3 and info["device"] == "cpu"


def test_actions_are_clipped_and_validated():
    env = KuramotoOscillatorEnv(n_oscillators=4, coupling_mode="constant")
    env.reset(seed=0)
    env.step(np.full(4, 10.0))
    np.testing.assert_array_equal(env.control_inputs, np.ones(4))
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(5))
    with pytest.raises(ValueError, match="non-finite"):
        env.step(np.array([0.0, np.nan, 0.0, 0.0]))


def test_step_and_render_before_reset_raise():
    env = KuramotoOscillatorEnv(n_oscillators=4, render_mode="rgb_array")
    with pytest.raises(ResetNeeded):
        env.step(env.action_space.sample())
    with pytest.raises(ResetNeeded):
        env.render()


def test_phase_history_is_bounded_by_episode_length():
    env = KuramotoOscillatorEnv(n_oscillators=4, max_steps=5, sync_threshold=2.0)
    env.reset(seed=0)
    for _ in range(20):
        env.step(env.action_space.sample())
    assert len(env.phase_history) == 6
    np.testing.assert_array_equal(env.phase_history[-1], env.phases)


# ---------------------------------------------------------------------------
# Seeding and global state
# ---------------------------------------------------------------------------
def rollout(env, seed, steps=15):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(99)
    observations, rewards = [obs], []
    for _ in range(steps):
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, *_ = env.step(action)
        observations.append(obs)
        rewards.append(reward)
    return np.array(observations), np.array(rewards)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_seeding_is_deterministic(backend):
    cls = KuramotoOscillatorEnv if backend == "numpy" else torch_env_class()
    env = cls(n_oscillators=6, noise_std=0.05)
    obs_a, rew_a = rollout(env, seed=3)
    obs_b, rew_b = rollout(env, seed=3)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(rew_a, rew_b)
    obs_c, _ = rollout(env, seed=4)
    assert not np.allclose(obs_a[0], obs_c[0])
    first, _ = env.reset()
    second, _ = env.reset()
    assert not np.allclose(first, second)


def test_no_global_rng_side_effects():
    torch = pytest.importorskip("torch")
    from env_lib.kos_env import KuramotoOscillatorEnvTorch

    # The global legacy RNG is inspected on purpose: the environments must not touch it.
    np_state = np.random.get_state()[1].copy()
    torch_state = torch.random.get_rng_state().clone()
    for cls in (KuramotoOscillatorEnv, KuramotoOscillatorEnvTorch):
        env = cls(n_oscillators=6, topology="random", noise_std=0.1)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        env.reset()
    np.testing.assert_array_equal(np.random.get_state()[1], np_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)


# ---------------------------------------------------------------------------
# Topologies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "topology, n_edges",
    [("fully_connected", 21), ("ring", 7), ("star", 6), ("random", None)],
)
def test_topologies(topology, n_edges):
    env = KuramotoOscillatorEnv(n_oscillators=7, topology=topology)
    matrix = env.topology_matrix
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), 0)
    assert set(np.unique(matrix)) <= {0.0, 1.0}
    assert env.n_couplings == int(matrix.sum() // 2)
    if n_edges is not None:
        assert env.n_couplings == n_edges
    assert env.action_space.shape == (7 + env.n_couplings,)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    np.testing.assert_array_equal(info["coupling_matrix"] != 0, matrix != 0)


def test_random_topology_matches_legacy_draw_and_is_seedable():
    rng = np.random.RandomState(42)
    legacy = (rng.random((8, 8)) > 0.5).astype(float)
    np.fill_diagonal(legacy, 0)
    legacy = ((legacy + legacy.T) > 0).astype(float)
    env = KuramotoOscillatorEnv(n_oscillators=8, topology="random")
    np.testing.assert_array_equal(env.topology_matrix, legacy)
    other = KuramotoOscillatorEnv(n_oscillators=8, topology="random", topology_seed=7)
    assert not np.array_equal(other.topology_matrix, legacy)


def test_custom_topology_and_edge_order():
    adjacency = np.array(
        [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
        dtype=float,
    )
    env = KuramotoOscillatorEnv(n_oscillators=4, topology="ring", adj_matrix=adjacency)
    assert env.topology == "custom"
    assert env.coupling_indices == [(0, 1), (0, 2), (1, 3), (2, 3)]
    matrix = env._coupling_matrix_from_actions(np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(matrix, matrix.T)
    assert matrix[0, 2] == 2.0 and matrix[3, 1] == 3.0 and matrix[0, 3] == 0.0
    rows, cols = topology_edges(build_topology("star", 5)[0])
    assert list(zip(rows.tolist(), cols.tolist())) == [(0, 1), (0, 2), (0, 3), (0, 4)]


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("normalize", [False, True])
def test_vectorised_dynamics_matches_naive_loop(normalize):
    rng = np.random.default_rng(0)
    n = 9
    env = KuramotoOscillatorEnv(n_oscillators=n, normalize_coupling=normalize)
    for _ in range(3):
        phases = rng.uniform(-np.pi, np.pi, n)
        freqs = rng.uniform(0.5, 2.0, n)
        coupling = rng.normal(size=(n, n))  # full matrix: negative and asymmetric entries count
        control = rng.uniform(-1, 1, n)
        np.testing.assert_allclose(
            env._kuramoto_dynamics(phases, freqs, coupling, control),
            naive_dynamics(phases, freqs, coupling, control, normalize),
            atol=1e-12,
        )


@pytest.mark.parametrize("normalize", [False, True])
def test_torch_dynamics_matches_naive_loop(normalize):
    torch = pytest.importorskip("torch")
    from env_lib.kos_env import KuramotoOscillatorEnvTorch

    rng = np.random.default_rng(1)
    n, batch = 7, 3
    env = KuramotoOscillatorEnvTorch(n_oscillators=n, n_agents=batch, normalize_coupling=normalize)
    phases = rng.uniform(-np.pi, np.pi, (batch, n))
    freqs = rng.uniform(0.5, 2.0, (batch, n))
    coupling = rng.normal(size=(batch, n, n))
    control = rng.uniform(-1, 1, (batch, n))
    result = env._kuramoto_dynamics(
        *(torch.as_tensor(x, dtype=torch.float32) for x in (phases, freqs, coupling, control))
    ).numpy()
    for b in range(batch):
        expected = naive_dynamics(phases[b], freqs[b], coupling[b], control[b], normalize)
        np.testing.assert_allclose(result[b], expected, atol=2e-5)

    strengths = torch.as_tensor(rng.uniform(0, 5, (batch, env.n_couplings)), dtype=torch.float32)
    batched = env._coupling_matrix_from_actions(strengths).numpy()
    reference = KuramotoOscillatorEnv(n_oscillators=n)
    for b in range(batch):
        expected = reference._coupling_matrix_from_actions(strengths[b].numpy().astype(np.float64))
        np.testing.assert_allclose(batched[b], expected, rtol=1e-6)


def test_rk4_is_more_accurate_than_euler():
    kwargs = dict(
        n_oscillators=6,
        dt=0.05,
        coupling_mode="constant",
        coupling_strength=0.8,
        sync_threshold=2.0,
        max_steps=100,
    )
    euler = KuramotoOscillatorEnv(integration_method="euler", **kwargs)
    rk4 = KuramotoOscillatorEnv(integration_method="rk4", **kwargs)
    euler.reset(seed=5)
    rk4.reset(seed=5)
    phases, freqs = euler.phases.copy(), euler.natural_frequencies.copy()
    coupling, control = euler.coupling_matrix, np.zeros(6)
    zero = np.zeros(6, dtype=np.float32)
    for _ in range(10):
        euler.step(zero)
        rk4.step(zero)
    fine_dt = 0.05 / 100  # reference: RK4 with a 100x smaller step
    for _ in range(1000):
        k1 = euler._kuramoto_dynamics(phases, freqs, coupling, control)
        k2 = euler._kuramoto_dynamics(phases + 0.5 * fine_dt * k1, freqs, coupling, control)
        k3 = euler._kuramoto_dynamics(phases + 0.5 * fine_dt * k2, freqs, coupling, control)
        k4 = euler._kuramoto_dynamics(phases + fine_dt * k3, freqs, coupling, control)
        phases = phases + fine_dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    error_euler = np.max(np.abs(angle_diff(euler.phases, phases)))
    error_rk4 = np.max(np.abs(angle_diff(rk4.phases, phases)))
    assert error_rk4 < 1e-4
    assert error_rk4 < 0.01 * error_euler


@pytest.mark.parametrize("integration_method", ["euler", "rk4"])
@pytest.mark.parametrize("coupling_mode", ["dynamic", "constant"])
@pytest.mark.parametrize("normalize", [False, True])
def test_numpy_and_torch_backends_agree(integration_method, coupling_mode, normalize):
    KuramotoOscillatorEnvTorch = torch_env_class()
    n = 6
    kwargs = dict(
        n_oscillators=n,
        integration_method=integration_method,
        coupling_mode=coupling_mode,
        normalize_coupling=normalize,
        coupling_strength=1.5,
        reward_type="combined",
        sync_threshold=2.0,
    )
    np_env = KuramotoOscillatorEnv(**kwargs)
    torch_env = KuramotoOscillatorEnvTorch(n_agents=2, **kwargs)
    rng = np.random.default_rng(11)
    options = {
        "phases": rng.uniform(-np.pi, np.pi, n),
        "natural_frequencies": rng.uniform(0.5, 2.0, n),
    }
    if coupling_mode == "dynamic":
        options["coupling_strengths"] = rng.uniform(0, 5, np_env.n_couplings)
    obs_np, info_np = np_env.reset(seed=0, options=options)
    obs_t, info_t = torch_env.reset(seed=1, options=options)
    np.testing.assert_allclose(obs_np, obs_t, atol=1e-6)
    np.testing.assert_allclose(info_np["phase_coherence"], info_t["phase_coherence"][0], atol=1e-5)
    for _ in range(8):
        action = rng.uniform(np_env.action_space.low, np_env.action_space.high).astype(np.float32)
        obs_np, rew_np, _, _, info_np = np_env.step(action)
        obs_t, rew_t, _, _, info_t = torch_env.step(action)
        assert np.max(np.abs(angle_diff(info_np["phases"], info_t["phases"][0]))) < 1e-5
        np.testing.assert_allclose(info_t["phases"][0], info_t["phases"][1])
        np.testing.assert_allclose(info_np["dphases_dt"], info_t["dphases_dt"][0], atol=1e-4)
        np.testing.assert_allclose(
            info_np["order_parameter"], info_t["order_parameter"][0], atol=1e-5
        )
        np.testing.assert_allclose(rew_np, rew_t, atol=1e-5)
        np.testing.assert_allclose(obs_np[n:], obs_t[n:], atol=1e-5)


def test_phase_coherence_definition_is_shared():
    KuramotoOscillatorEnvTorch = torch_env_class()
    import torch

    phases = np.array([[3.0, -3.0, 0.1, 2.9], [0.0, 0.2, -0.2, 0.1]])
    expected = np.exp(-np.var((phases + np.pi) % (2 * np.pi) - np.pi, axis=1))
    env_np = KuramotoOscillatorEnv(n_oscillators=4)
    env_t = KuramotoOscillatorEnvTorch(n_oscillators=4, n_agents=2)
    for row, value in zip(phases, expected):
        assert env_np._compute_phase_coherence(row) == pytest.approx(value)
    torch_values = env_t._compute_phase_coherence(torch.as_tensor(phases, dtype=torch.float32))
    np.testing.assert_allclose(torch_values.numpy(), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Coupling modes
# ---------------------------------------------------------------------------
def test_constant_mode_without_matrix_uses_coupling_strength():
    env = KuramotoOscillatorEnv(n_oscillators=5, coupling_mode="constant", topology="ring")
    np.testing.assert_array_equal(env.coupling_matrix, env.topology_matrix)
    env = KuramotoOscillatorEnv(
        n_oscillators=5, coupling_mode="constant", topology="ring", coupling_strength=2.5
    )
    np.testing.assert_array_equal(env.coupling_matrix, 2.5 * env.topology_matrix)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(info["coupling_matrix"], 2.5 * env.topology_matrix)


def test_constant_mode_with_matrix():
    matrix = np.arange(16, dtype=float).reshape(4, 4)
    env = KuramotoOscillatorEnv(
        n_oscillators=4, coupling_mode="constant", constant_coupling_matrix=matrix
    )
    expected = matrix.copy()
    np.fill_diagonal(expected, 0)  # self-coupling has no effect and is dropped
    np.testing.assert_array_equal(env.coupling_matrix, expected)
    assert env.action_space.shape == (4,) and env.observation_space.shape == (12,)
    env.reset(seed=0)
    phases, freqs = env.phases.copy(), env.natural_frequencies.copy()
    _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(
        info["dphases_dt"], naive_dynamics(phases, freqs, matrix, np.zeros(4)), atol=1e-12
    )
    with pytest.warns(UserWarning, match="ignored"):
        KuramotoOscillatorEnv(n_oscillators=4, constant_coupling_matrix=matrix)


def test_torch_constant_mode_with_and_without_matrix():
    KuramotoOscillatorEnvTorch = torch_env_class()
    env = KuramotoOscillatorEnvTorch(n_oscillators=5, n_agents=2, coupling_mode="constant")
    assert tuple(env.coupling_matrix.shape) == (2, 5, 5)
    np.testing.assert_array_equal(env.coupling_matrix[0].numpy(), 1.0 - np.eye(5))
    matrix = np.full((5, 5), 0.5)
    env = KuramotoOscillatorEnvTorch(
        n_oscillators=5, coupling_mode="constant", constant_coupling_matrix=matrix
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    assert info["coupling_matrix"].shape == (1, 5, 5)
    np.testing.assert_allclose(info["coupling_matrix"][0], matrix - 0.5 * np.eye(5))


# ---------------------------------------------------------------------------
# Rewards, termination and truncation
# ---------------------------------------------------------------------------
def test_truncation_at_step_limit():
    env = KuramotoOscillatorEnv(
        n_oscillators=6, max_steps=3, coupling_mode="constant", coupling_strength=0.0
    )
    env.reset(seed=0, options=spread_state(6))
    flags = [env.step(np.zeros(6, dtype=np.float32))[2:4] for _ in range(3)]
    assert flags == [(False, False), (False, False), (False, True)]


@pytest.mark.parametrize(
    "reward_type, bonus_applied",
    [
        ("order_parameter", True),
        ("phase_coherence", True),
        ("combined", True),
        ("frequency_synchronization", False),
    ],
)
def test_synchronisation_terminates_with_bonus(reward_type, bonus_applied):
    n = 5
    env = KuramotoOscillatorEnv(
        n_oscillators=n, coupling_mode="constant", reward_type=reward_type, sync_bonus=3.0
    )
    env.reset(seed=0, options=synced_state(n))
    _, reward, terminated, truncated, info = env.step(np.zeros(n, dtype=np.float32))
    assert terminated is True and truncated is False
    r, coherence = info["order_parameter"], info["phase_coherence"]
    base = {
        "order_parameter": r,
        "phase_coherence": coherence,
        "combined": r + coherence,
        "frequency_synchronization": -np.mean(np.abs(info["dphases_dt"] - 1.0)),
    }[reward_type]
    assert reward == pytest.approx(base + (3.0 if bonus_applied else 0.0))

    # Not synchronised: no bonus, no termination.
    env.reset(seed=0, options=spread_state(n))
    _, reward, terminated, _, info = env.step(np.zeros(n, dtype=np.float32))
    assert terminated is False and info["order_parameter"] < 0.5
    if reward_type == "order_parameter":
        assert reward == pytest.approx(info["order_parameter"])


def test_sync_threshold_above_one_disables_termination():
    env = KuramotoOscillatorEnv(n_oscillators=4, coupling_mode="constant", sync_threshold=1.5)
    env.reset(seed=0, options=synced_state(4))
    _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert terminated is False
    assert reward == pytest.approx(info["order_parameter"])


def test_torch_rewards_termination_and_bonus_per_system():
    KuramotoOscillatorEnvTorch = torch_env_class()
    n = 5
    env = KuramotoOscillatorEnvTorch(
        n_oscillators=n, n_agents=2, coupling_mode="constant", max_steps=2
    )
    options = {
        "phases": np.stack([synced_state(n)["phases"], spread_state(n)["phases"]]),
        "natural_frequencies": np.full(n, 1.0),
    }
    env.reset(seed=0, options=options)
    _, reward, terminated, truncated, info = env.step(np.zeros(n, dtype=np.float32))
    rewards = info["agent_rewards"]
    assert rewards.shape == (2,) and rewards.dtype == np.float64
    assert terminated is True and truncated is False  # decided on system 0
    assert reward == pytest.approx(rewards[0])
    assert rewards[0] == pytest.approx(info["order_parameter"][0] + 10.0, abs=1e-5)
    assert rewards[1] == pytest.approx(info["order_parameter"][1], abs=1e-5)
    np.testing.assert_allclose(env.get_batch_rewards().numpy(), info["order_parameter"], atol=1e-6)
    np.testing.assert_allclose(
        env.get_batch_rewards(include_bonus=True).numpy(), rewards, atol=1e-5
    )
    assert env.step(np.zeros(n, dtype=np.float32))[3] is True

    freq_env = KuramotoOscillatorEnvTorch(
        n_oscillators=n, coupling_mode="constant", reward_type="frequency_synchronization"
    )
    freq_env.reset(seed=0, options=synced_state(n))
    _, reward, terminated, _, info = freq_env.step(np.zeros(n, dtype=np.float32))
    assert terminated is True
    assert reward == pytest.approx(-np.mean(np.abs(info["dphases_dt"][0] - 1.0)), abs=1e-6)


# ---------------------------------------------------------------------------
# PyTorch batch API and device handling
# ---------------------------------------------------------------------------
def test_torch_batch_api():
    KuramotoOscillatorEnvTorch = torch_env_class()
    import torch

    env = KuramotoOscillatorEnvTorch(n_oscillators=4, n_agents=3, device="auto")
    assert env.device.type in ("cpu", "cuda")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    batch = env.get_batch_observations()
    assert tuple(batch.shape) == (3, env.observation_space.shape[0])
    np.testing.assert_allclose(batch[0].cpu().numpy(), obs)
    assert info["phases"].shape == (3, 4) and info["coupling_matrix"].shape == (3, 4, 4)

    actions = np.stack([env.action_space.sample() for _ in range(3)])
    obs, _, _, _, info = env.step(actions)
    np.testing.assert_allclose(
        env.control_inputs.cpu().numpy(), np.clip(actions[:, :4], -1, 1), atol=1e-6
    )
    assert env.get_batch_rewards().shape == (3,)
    assert info["dphases_dt"].shape == (3, 4)
    env.step(torch.zeros(env.action_space.shape))
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros((2, env.action_space.shape[0])))
    info["phases"][:] = 99.0  # info arrays are copies
    assert float(env.phases.abs().max()) <= np.pi
    with pytest.raises(ValueError, match="render_agent"):
        KuramotoOscillatorEnvTorch(n_agents=2, render_agent=2)


# ---------------------------------------------------------------------------
# Validation and backwards compatibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"topology": "hexagon"},
        {"topology": "custom"},
        {"integration_method": "midpoint"},
        {"reward_type": "entropy"},
        {"coupling_mode": "static"},
        {"adj_matrix": np.ones((3, 3))},
        {"coupling_mode": "constant", "constant_coupling_matrix": np.ones((4, 5))},
        {"n_oscillators": 1},
        {"n_agents": 0},
        {"dt": 0.0},
        {"max_steps": 0},
        {"noise_std": -0.1},
        {"coupling_range": (2.0, 1.0)},
        {"natural_freq_range": (1.0,)},
        {"sync_threshold": 0.0},
        {"topology_seed": -1},
        {"render_mode": "ascii"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        KuramotoOscillatorEnv(**kwargs)


def test_invalid_argument_types_raise_type_error():
    with pytest.raises(TypeError):
        KuramotoOscillatorEnv(n_oscillators=4.5)
    with pytest.raises(TypeError):
        KuramotoOscillatorEnv(normalize_coupling="yes")


def test_torch_rejects_bad_device_and_matrix():
    KuramotoOscillatorEnvTorch = torch_env_class()
    with pytest.raises(ValueError):
        KuramotoOscillatorEnvTorch(device="not-a-device")
    with pytest.raises(ValueError):
        KuramotoOscillatorEnvTorch(n_oscillators=4, adj_matrix=np.ones((5, 5)))


def test_reset_options_are_validated():
    env = KuramotoOscillatorEnv(n_oscillators=4, coupling_mode="constant")
    with pytest.raises(ValueError, match="Unknown reset option"):
        env.reset(options={"velocity": np.zeros(4)})
    with pytest.raises(ValueError, match="shape"):
        env.reset(options={"phases": np.zeros(3)})
    with pytest.raises(ValueError, match="dynamic"):
        env.reset(options={"coupling_strengths": np.zeros(6)})
    obs, _ = env.reset(options={"phases": np.full(4, 4.0)})
    np.testing.assert_allclose(obs[:4], 4.0 - 2 * np.pi, atol=1e-6)


def test_deprecated_register_function():
    from env_lib import kos_env

    with pytest.warns(DeprecationWarning, match="register_envs"):
        kos_env.register()


# Gymnasium flags e.g. "KuramotoOscillator-v0" as out of date because a "-v1" id exists.
@pytest.mark.filterwarnings("ignore:.*is out of date:DeprecationWarning")
@pytest.mark.parametrize("env_id", KURAMOTO_IDS)
def test_make_registered_ids(env_id):
    if "Torch" in env_id:
        pytest.importorskip("torch")
    env = env_lib.make(env_id)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and type(terminated) is bool and type(truncated) is bool
    assert info["agent_rewards"].shape == (env.unwrapped.n_agents,)
    env.close()


# The unbounded observation Box and the [0, 5] coupling action range are part of the
# public API; check_env only flags them as advisory warnings.
@pytest.mark.filterwarnings("ignore:.*symmetric and normalized space:UserWarning")
@pytest.mark.filterwarnings("ignore:.*observation space m.* value is .*infinity:UserWarning")
@pytest.mark.parametrize("coupling_mode", ["dynamic", "constant"])
def test_check_env(coupling_mode):
    env = KuramotoOscillatorEnv(n_oscillators=5, coupling_mode=coupling_mode)
    check_env(env.unwrapped, skip_render_check=True)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_rgb_array_frames(backend, theme):
    cls = KuramotoOscillatorEnv if backend == "numpy" else torch_env_class()
    previous = rendering.get_theme()
    rendering.set_theme(theme)
    try:
        env = cls(n_oscillators=8, render_mode="rgb_array")
        env.reset(seed=0)
        first = env.render()
        assert first.shape == (560, 1000, 3) and first.dtype == np.uint8
        assert first.std() > 5  # not a constant image
        expected_background = np.array(
            [int(rendering.get_theme(theme).background[i : i + 2], 16) for i in (1, 3, 5)]
        )
        np.testing.assert_array_equal(first[2, 2], expected_background)
        for _ in range(3):
            env.step(env.action_space.sample())
        second = env.render()
        assert second.shape == first.shape and not np.array_equal(first, second)
        env.reset(seed=1)
        assert env.render().shape == first.shape
        env.close()
        env.close()
        assert env.render().shape == first.shape  # re-created after close
        env.close()
    finally:
        rendering.set_theme(previous)


def test_render_without_mode_warns_once_and_returns_none():
    env = KuramotoOscillatorEnv(n_oscillators=4)
    env.reset(seed=0)
    with pytest.warns(UserWarning, match="render_mode"):
        assert env.render() is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert env.render() is None


def test_human_mode_is_headless_safe():
    env = KuramotoOscillatorEnv(n_oscillators=4, render_mode="human")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # non-interactive backend
        env.reset(seed=0)
        env.step(env.action_space.sample())
        assert env.render() is None
    env.close()
    env.close()


def test_close_is_idempotent_without_render():
    env = KuramotoOscillatorEnv(n_oscillators=4, render_mode="rgb_array")
    env.close()
    env.reset(seed=0)
    env.close()
    env.close()
