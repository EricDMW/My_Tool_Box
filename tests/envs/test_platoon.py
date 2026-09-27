"""Tests for :mod:`env_lib.platoon_env` (vehicle platoon, CACC)."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env

import env_lib
from env_lib.errors import ResetNeededError
from env_lib.platoon_env import (
    FEATURE_SCALES,
    OBSERVATION_FEATURES,
    SCENARIOS,
    TOPOLOGIES,
    PlatoonConfig,
    PlatoonEnv,
    PlatoonVectorEnv,
    cacc_policy,
    platoon_adjacency,
    string_stability_gain,
)
from env_lib.utils import rendering
from env_lib.utils.rendering import figure_to_rgb

N = 8
QUIET = {"init_spacing_noise": 0.0, "init_speed_noise": 0.0}


def acc_policy(obs):
    return cacc_policy(obs, k_a=0.0)


def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        if terminated or truncated:
            return total, info


def state_of(env):
    """Full initial state of a single environment as reset options."""
    return {
        "positions": env.positions,
        "velocities": env.velocities,
        "accelerations": env.accelerations,
        "leader_command": env.leader_command,
    }


def reference_step(env, action):
    """One step of every vehicle with the matrix exponential of the continuous model."""
    from scipy.linalg import expm

    command = np.concatenate([[env.leader_command[env.step_count]], action])
    state = np.stack([env.positions, env.velocities, env.accelerations], axis=1)
    out = np.empty_like(state)
    for i, tau in enumerate(env.time_constants):
        augmented = np.zeros((4, 4))
        augmented[0, 1] = augmented[1, 2] = 1.0
        augmented[2, 2] = -1.0 / tau
        augmented[2, 3] = 1.0 / tau
        transition = expm(augmented * env.dt)
        out[i] = transition[:3, :3] @ state[i] + transition[:3, 3] * command[i]
    return out


def reference_observation(env, last_commands):
    """Observation rows built follower by follower from the physical state."""
    p, v, a, lengths = env.positions, env.velocities, env.accelerations, env.lengths
    h, r = env.headway, env.standstill_distance
    topology, n = env.topology, env.n_followers
    scale = FEATURE_SCALES
    rows = []
    for i in range(1, n + 1):
        error = p[i - 1] - p[i] - lengths[i - 1] - r - h * v[i]
        row = dict.fromkeys(OBSERVATION_FEATURES, 0.0)
        row["spacing_error"] = error / scale["spacing_error"]
        row["relative_speed"] = (v[i - 1] - v[i]) / scale["relative_speed"]
        row["speed"] = v[i] / scale["speed"]
        row["acceleration"] = a[i] / scale["acceleration"]
        if topology != "none":
            row["predecessor_available"] = 1.0
            row["predecessor_acceleration"] = a[i - 1] / scale["acceleration"]
            row["predecessor_command"] = last_commands[i - 1] / scale["acceleration"]
        if topology == "predecessor_leader":
            row["leader_available"] = 1.0
            row["leader_speed_error"] = (v[0] - v[i]) / scale["leader_speed_error"]
            row["leader_acceleration_error"] = (a[0] - a[i]) / scale["acceleration"]
        if topology == "bidirectional" and i < n:
            follower_error = p[i] - p[i + 1] - lengths[i] - r - h * v[i + 1]
            row["follower_available"] = 1.0
            row["follower_spacing_error"] = follower_error / scale["spacing_error"]
            row["follower_relative_speed"] = (v[i] - v[i + 1]) / scale["relative_speed"]
        rows.append([row[name] for name in OBSERVATION_FEATURES])
    return np.array(rows, dtype=np.float32)


@pytest.fixture
def light_theme():
    previous = rendering.get_theme()
    rendering.set_theme("light")
    try:
        yield
    finally:
        rendering.set_theme(previous)


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------
def test_default_spaces_and_step_types():
    env = PlatoonEnv()
    assert env.action_space == Box(-6.0, 3.0, shape=(N,), dtype=np.float32)
    assert env.observation_space.shape == (N, len(OBSERVATION_FEATURES)) == (N, 13)
    assert env.observation_space.dtype == np.float32
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)
    assert np.all(info["agent_rewards"] == 0.0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs) and np.all(np.isfinite(obs))
    assert type(reward) is float and type(terminated) is bool and type(truncated) is bool
    assert info["agent_rewards"].shape == (N,) and info["agent_rewards"].dtype == np.float64
    assert reward == pytest.approx(info["agent_rewards"].sum())
    for key in ("spacing_errors", "gaps", "relative_speeds", "speeds", "peak_spacing_errors"):
        assert info[key].shape == (N,)
    for key in ("leader_speed", "min_gap", "error_amplification", "time"):
        assert isinstance(info[key], float)
    assert isinstance(info["collision"], bool) and isinstance(info["breakup"], bool)
    assert info["step"] == 1 and info["min_gap"] == pytest.approx(info["gaps"].min())


def test_check_env():
    for topology in ("predecessor_leader", "bidirectional"):
        env = PlatoonEnv(topology=topology, randomize_vehicles=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # The observation space is deliberately unbounded (spacing errors, relative speeds).
            warnings.filterwarnings("ignore", message=r".*Box observation space m\w+ value is")
            # Actions are accelerations in m/s^2 with physical, asymmetric bounds [-6, 3].
            warnings.filterwarnings("ignore", message=r".*symmetric and normalized space")
            check_env(env.unwrapped, skip_render_check=True)


def test_make_and_make_vec_use_native_classes():
    env = env_lib.make("Platoon-v0")
    assert isinstance(env.unwrapped, PlatoonEnv)
    obs, _ = env.reset(seed=0)
    env.step(cacc_policy(obs))
    env.close()
    envs = env_lib.make_vec("Platoon-v0", 8)
    assert isinstance(envs, PlatoonVectorEnv) and envs.num_envs == 8
    obs, _ = envs.reset(seed=0)
    assert obs.shape == (8, N, 13)
    envs.close()
    assert env_lib.PlatoonEnv is PlatoonEnv and env_lib.PlatoonVectorEnv is PlatoonVectorEnv


def test_config_defaults_match_constructor():
    signature = inspect.signature(PlatoonEnv.__init__).parameters
    for name, field in PlatoonConfig.__dataclass_fields__.items():
        assert signature[name].default == field.default, name
    assert PlatoonEnv().config == PlatoonConfig()


def test_seeding_determinism():
    env_a, env_b = PlatoonEnv(randomize_vehicles=True), PlatoonEnv(randomize_vehicles=True)
    obs_a, _ = env_a.reset(seed=3)
    obs_b, _ = env_b.reset(seed=3)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(env_a.leader_command, env_b.leader_command)
    rng = np.random.default_rng(0)
    for _ in range(30):
        action = cacc_policy(obs_a) + rng.normal(0.0, 0.3, N).astype(np.float32)
        obs_a, reward_a, *_ = env_a.step(action)
        obs_b, reward_b, *_ = env_b.step(action)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert reward_a == reward_b
    env_a.reset(seed=4)
    assert not np.array_equal(env_a.leader_command, env_b.leader_command)
    assert not np.array_equal(env_a.time_constants, env_b.time_constants)


def test_fixed_vehicles_come_from_vehicle_seed():
    a, b = PlatoonEnv(vehicle_seed=5), PlatoonEnv(vehicle_seed=5)
    np.testing.assert_array_equal(a.time_constants, b.time_constants)
    a.reset(seed=0)
    first = a.time_constants
    a.reset(seed=1)
    np.testing.assert_array_equal(a.time_constants, first)
    assert np.all((first >= 0.2) & (first <= 0.4))
    assert np.all((a.lengths >= 4.0) & (a.lengths <= 5.0))
    assert not np.array_equal(PlatoonEnv(vehicle_seed=6).time_constants, first)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def test_step_is_exact_discretisation_of_the_lagged_model():
    env = PlatoonEnv(randomize_vehicles=True)
    env.reset(seed=2)
    rng = np.random.default_rng(1)
    for _ in range(20):
        action = rng.uniform(-2.0, 2.0, N).astype(np.float32)
        expected = reference_step(env, action.astype(np.float64))
        env.step(action)
        np.testing.assert_allclose(env.positions, expected[:, 0], rtol=0, atol=1e-9)
        np.testing.assert_allclose(env.velocities, expected[:, 1], rtol=0, atol=1e-9)
        np.testing.assert_allclose(env.accelerations, expected[:, 2], rtol=0, atol=1e-12)


def test_initial_state_is_perturbed_equilibrium():
    env = PlatoonEnv(**QUIET)
    _, info = env.reset(seed=0)
    np.testing.assert_allclose(info["spacing_errors"], 0.0, atol=1e-12)
    np.testing.assert_allclose(info["speeds"], info["leader_speed"])
    assert env.positions[0] == 0.0 and np.all(np.diff(env.positions) < 0)
    assert 16.0 <= info["leader_speed"] <= 25.0
    env = PlatoonEnv()
    _, info = env.reset(seed=0)
    # Perturbations are truncated at three standard deviations.
    bound = 3.0 * (env.config.init_spacing_noise + env.headway * env.config.init_speed_noise)
    assert 0.0 < np.abs(info["spacing_errors"]).max() <= bound + 1e-12


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_constant_speed_leader_and_cacc_stay_at_equilibrium(topology):
    env = PlatoonEnv(topology=topology, max_steps=100, **QUIET)
    obs, info = env.reset(seed=1, options={"leader_command": np.zeros(100)})
    start, total = env.velocities, 0.0
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(cacc_policy(obs))
        total += reward
    assert truncated and not terminated
    np.testing.assert_allclose(info["spacing_errors"], 0.0, atol=1e-9)
    np.testing.assert_allclose(env.velocities, start, atol=1e-12)
    assert abs(total) < 1e-12


def test_observation_matches_reference():
    for topology in TOPOLOGIES:
        env = PlatoonEnv(topology=topology, scenario="random", randomize_vehicles=True)
        obs, _ = env.reset(seed=7)
        np.testing.assert_allclose(obs, reference_observation(env, np.zeros(N + 1)), atol=1e-6)
        for _ in range(15):
            obs, *_, info = env.step(cacc_policy(obs))
            commands = np.concatenate([[env.leader_command[env.step_count - 1]], info["commands"]])
            np.testing.assert_allclose(obs, reference_observation(env, commands), atol=1e-6)
        layout = env.observation_layout
        assert list(layout) == list(OBSERVATION_FEATURES)
        assert obs[:, layout["speed"]].ravel() == pytest.approx(env.velocities[1:] / 30.0)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_topology_sets_links_and_availability_flags(topology):
    env = PlatoonEnv(topology=topology)
    obs, _ = env.reset(seed=0)
    layout = env.observation_layout
    flags = {name: obs[:, layout[f"{name}_available"]].ravel() for name in ("predecessor", "leader", "follower")}  # fmt: skip
    adjacency, leader_links = env.adjacency, env.leader_links
    np.testing.assert_array_equal(adjacency, platoon_adjacency(topology, N)[0])
    assert adjacency.shape == (N, N) and adjacency.dtype == bool and leader_links.shape == (N,)
    predecessor = np.eye(N, k=-1, dtype=bool)
    expected = {
        "predecessor": (predecessor, [1] + [0] * (N - 1), 1, 0, [0] * N),
        "predecessor_leader": (predecessor, [1] * N, 1, 1, [0] * N),
        "bidirectional": (predecessor | predecessor.T, [1] + [0] * (N - 1), 1, 0, [1] * (N - 1) + [0]),
        "none": (np.zeros((N, N), bool), [0] * N, 0, 0, [0] * N),
    }[topology]  # fmt: skip
    np.testing.assert_array_equal(adjacency, expected[0])
    np.testing.assert_array_equal(leader_links, expected[1])
    assert np.all(flags["predecessor"] == expected[2]) and np.all(flags["leader"] == expected[3])
    np.testing.assert_array_equal(flags["follower"], expected[4])
    # Unavailable channels are zero-padded.
    for name in ("predecessor", "leader", "follower"):
        columns = [k for k, f in enumerate(OBSERVATION_FEATURES) if f.startswith(name)]
        assert np.all(obs[flags[name] == 0][:, columns] == 0.0)


def test_reward_matches_reference():
    env = PlatoonEnv(scenario="stop_and_go", jerk_weight=0.02)
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(3)
    for _ in range(40):
        before = env.accelerations[1:]
        action = cacc_policy(obs) + rng.normal(0.0, 0.5, N).astype(np.float32)
        obs, reward, terminated, _, info = env.step(action)
        u = np.clip(action.astype(np.float64), -6.0, 3.0)
        jerk = (env.accelerations[1:] - before) / env.dt
        e = (
            env.positions[:-1]
            - env.positions[1:]
            - env.lengths[:-1]
            - 2.0
            - 0.6 * env.velocities[1:]
        )
        dv = env.velocities[:-1] - env.velocities[1:]
        expected = -(e**2 + 0.1 * dv**2 + 0.02 * u**2 + 0.02 * jerk**2) * env.dt
        np.testing.assert_allclose(info["agent_rewards"], expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(info["commands"], u)
        assert reward == pytest.approx(expected.sum()) and not terminated


def _inject(env, gaps, speeds, **options):
    """Positions from gaps (m) and speeds (leader first) for reset options."""
    lengths = env.lengths
    positions = np.zeros(N + 1)
    positions[1:] = -np.cumsum(np.asarray(gaps) + lengths[:-1])
    return {"positions": positions, "velocities": np.asarray(speeds, float), **options}


def test_collision_terminates_with_penalty():
    env = PlatoonEnv(max_steps=50)
    gaps = [15.0] * N
    gaps[3] = 0.5
    speeds = np.full(N + 1, 20.0)
    speeds[4] = 30.0  # follower 4 closes a 0.5 m gap at 10 m/s
    env.reset(seed=0, options=_inject(env, gaps, speeds, leader_command=np.zeros(50)))
    _, reward, terminated, truncated, info = env.step(np.zeros(N))
    assert terminated and not truncated and info["collision"] and not info["breakup"]
    assert info["min_gap"] <= 0.0 and info["gaps"][3] <= 0.0
    assert info["agent_rewards"][3] < -env.config.collision_penalty
    assert np.all(info["agent_rewards"][np.arange(N) != 3] > -10.0)
    assert reward == pytest.approx(info["agent_rewards"].sum())


def test_breakup_terminates_and_vehicles_do_not_reverse():
    env = PlatoonEnv(max_steps=50)
    gaps = [15.0] * N
    gaps[0] = 99.5
    speeds = np.full(N + 1, 20.0)
    speeds[1:] = 0.0
    env.reset(seed=0, options=_inject(env, gaps, speeds, leader_command=np.zeros(50)))
    _, _, terminated, _, info = env.step(np.full(N, -6.0))
    assert terminated and info["breakup"] and not info["collision"]
    assert info["agent_rewards"][0] < -env.config.breakup_penalty
    np.testing.assert_array_equal(info["speeds"], 0.0)  # full braking at standstill
    assert np.all(info["accelerations"] >= 0.0)


def test_truncation_and_leader_profiles():
    env = PlatoonEnv(max_steps=5)
    env.reset(seed=0)
    for step in range(1, 6):
        *_, terminated, truncated, info = env.step(np.zeros(N))
        assert truncated == (step == 5) and info["step"] == step
    for scenario in SCENARIOS:
        env = PlatoonEnv(scenario=scenario)
        for seed in range(3):
            env.reset(seed=seed)
            command = env.leader_command
            assert command.shape == (600,) and np.all(command >= -4.0) and np.all(command <= 2.5)
            reference = env.velocities[0] + np.cumsum(command) * env.dt
            assert reference.min() >= -1e-9 and reference.max() <= 30.0 + 1e-9
            if scenario in ("stop_and_go", "mixed"):
                assert reference.min() <= 4.0 + 1e-9  # braking to low speed
            if scenario == "cruise":
                assert np.ptp(reference) <= 4.0 + 1e-9


# ---------------------------------------------------------------------------
# Baselines and string stability
# ---------------------------------------------------------------------------
def test_cacc_policy_shapes_and_acc_fallback():
    env = PlatoonEnv()
    obs, _ = env.reset(seed=0)
    action = cacc_policy(obs)
    assert action.shape == (N,) and action.dtype == np.float32 and env.action_space.contains(action)
    batch = np.stack([obs, obs * 2.0, obs])
    np.testing.assert_allclose(cacc_policy(batch)[0], action)
    assert cacc_policy(batch.reshape(3, 1, N, 13)).shape == (3, 1, N)
    no_v2v = obs.copy()
    no_v2v[:, env.observation_layout["predecessor_available"]] = 0.0
    np.testing.assert_allclose(cacc_policy(no_v2v), acc_policy(obs))
    with pytest.raises(ValueError):
        cacc_policy(np.zeros((N, 12)))
    with pytest.raises(ValueError):
        cacc_policy(np.zeros(13))


def test_cacc_beats_random_actions():
    env = PlatoonEnv()
    env.action_space.seed(0)
    cacc = [rollout(env, cacc_policy, seed) for seed in range(3)]
    random = [rollout(env, lambda _obs: env.action_space.sample(), seed) for seed in range(3)]
    assert all(not info["collision"] and info["step"] == env.max_steps for _, info in cacc)
    assert max(ret for ret, _ in cacc) > 10.0 * max(ret for ret, _ in random)
    assert min(ret for ret, _ in cacc) > -150.0


def test_cacc_is_string_stable_and_short_headway_acc_is_not():
    # Frequency domain: peak gain of the string-stability transfer function.
    for tau in (0.2, 0.3, 0.4):
        assert string_stability_gain(time_constant=tau) <= 1.0 + 1e-9
        assert string_stability_gain(k_a=0.0, time_constant=tau) > 1.2
        assert string_stability_gain(k_a=0.0, headway=1.2, time_constant=tau) <= 1.0 + 1e-9
    assert string_stability_gain(k_p=50.0, k_d=0.0, k_a=0.0) == np.inf  # unstable own loop
    # Time domain: a speed wave travels down the platoon.
    _, info = rollout(PlatoonEnv(scenario="cruise", **QUIET), acc_policy, 0)
    assert not info["collision"] and info["error_amplification"] > 1.5
    peaks = info["peak_spacing_errors"]
    assert np.all(np.diff(peaks) > 0.0)  # every follower worse than its predecessor
    _, info = rollout(PlatoonEnv(scenario="cruise", **QUIET), cacc_policy, 0)
    assert info["error_amplification"] < 1.0
    _, info = rollout(PlatoonEnv(scenario="stop_and_go", **QUIET), acc_policy, 0)
    assert info["collision"]
    _, info = rollout(PlatoonEnv(scenario="stop_and_go", **QUIET), cacc_policy, 0)
    assert not info["collision"] and info["error_amplification"] < 0.6
    assert info["min_gap"] > 1.0


# ---------------------------------------------------------------------------
# Vector environment
# ---------------------------------------------------------------------------
def test_vector_shapes_and_infos():
    envs = PlatoonVectorEnv(5, n_followers=4, topology="bidirectional")
    assert envs.observation_space.shape == (5, 4, 13) and envs.action_space.shape == (5, 4)
    assert envs.single_observation_space == PlatoonEnv(n_followers=4).observation_space
    obs, infos = envs.reset(seed=0)
    assert obs.shape == (5, 4, 13) and obs.dtype == np.float32 and obs.flags["C_CONTIGUOUS"]
    obs, rewards, terminated, truncated, infos = envs.step(cacc_policy(obs))
    assert rewards.shape == terminated.shape == truncated.shape == (5,)
    assert infos["agent_rewards"].shape == (5, 4) and infos["_agent_rewards"].all()
    np.testing.assert_allclose(rewards, infos["agent_rewards"].sum(axis=1))
    assert infos["min_gap"].shape == (5,) and infos["step"].tolist() == [1] * 5
    assert envs.adjacency.shape == (4, 4) and envs.n_agents == 4
    with pytest.raises(ValueError):
        envs.step(np.full((5, 4), np.nan))
    with pytest.raises(ValueError):
        envs.step(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        PlatoonVectorEnv(2, n_followers=0)
    with pytest.raises(TypeError):
        PlatoonVectorEnv(2, followers=3)


def test_vector_copy_zero_matches_single_env():
    kwargs = {"topology": "predecessor_leader", "scenario": "stop_and_go"}
    single = PlatoonEnv(**kwargs)
    obs_s, _ = single.reset(seed=5)
    envs = PlatoonVectorEnv(3, **kwargs)
    obs_v, _ = envs.reset(seed=0, options=state_of(single))
    np.testing.assert_array_equal(obs_v[0], obs_s)
    np.testing.assert_array_equal(obs_v[1], obs_s)  # unbatched options apply to every copy
    rng = np.random.default_rng(0)
    for _ in range(120):
        actions = cacc_policy(obs_v) + rng.normal(0.0, 0.4, (3, N)).astype(np.float32)
        obs_s, reward_s, term_s, trunc_s, info_s = single.step(actions[0])
        obs_v, rewards, terminated, truncated, infos = envs.step(actions)
        np.testing.assert_array_equal(obs_v[0], obs_s)
        assert rewards[0] == pytest.approx(reward_s, rel=1e-12, abs=1e-12)
        assert terminated[0] == term_s and truncated[0] == trunc_s
        for key in ("agent_rewards", "spacing_errors", "peak_spacing_errors", "min_gap"):
            np.testing.assert_allclose(infos[key][0], info_s[key], rtol=1e-12, atol=1e-12)


def test_vector_autoreset_and_per_copy_options():
    envs = PlatoonVectorEnv(3, max_steps=4, scenario="cruise")
    single = PlatoonEnv(max_steps=4)
    gaps = [15.0] * N
    gaps[0] = 0.5
    speeds = np.full(N + 1, 20.0)
    speeds[1] = 30.0
    crash = _inject(single, gaps, speeds)
    safe = _inject(single, [14.0] * N, np.full(N + 1, 20.0))
    options = {key: np.stack([crash[key], safe[key], safe[key]]) for key in crash}
    obs, _ = envs.reset(seed=0, options=options)
    obs, rewards, terminated, truncated, infos = envs.step(np.zeros((3, N)))
    assert terminated.tolist() == [True, False, False] and infos["collision"].tolist()[0]
    obs, rewards, terminated, truncated, infos = envs.step(np.zeros((3, N)))
    # Copy 0 was reset (next-step autoreset): zero reward, fresh state.
    assert rewards[0] == 0.0 and not terminated[0]
    assert infos["step"].tolist() == [0, 2, 2] and not infos["collision"][0]
    assert infos["agent_rewards"][0].tolist() == [0.0] * N
    assert envs._core.steps.tolist() == [0, 2, 2]
    assert np.all(envs._core.gap[0] > 1.0)
    for _ in range(2):
        *_, truncated, _ = envs.step(np.zeros((3, N)))
    assert truncated.tolist() == [False, True, True]

    same = PlatoonVectorEnv(2, autoreset_mode="same_step", max_steps=2)
    same.reset(seed=0)
    same.step(np.zeros((2, N)))
    obs, _, _, truncated, infos = same.step(np.zeros((2, N)))
    assert truncated.all() and infos["_final_obs"].all()
    assert infos["final_obs"][0].shape == (N, 13) and same._core.steps.tolist() == [0, 0]


def test_vector_randomized_vehicles_differ_between_copies():
    envs = PlatoonVectorEnv(4, randomize_vehicles=True)
    envs.reset(seed=0)
    tau = envs._core.tau
    assert not np.array_equal(tau[0], tau[1])
    fixed = PlatoonVectorEnv(4)
    fixed.reset(seed=0)
    np.testing.assert_array_equal(fixed._core.tau[0], PlatoonEnv().time_constants)
    assert np.all(fixed._core.tau == fixed._core.tau[0])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_followers": 0},
        {"topology": "ring"},
        {"scenario": "highway"},
        {"dt": 0.0},
        {"max_steps": 0},
        {"headway": -0.1},
        {"standstill_distance": 0.0},
        {"accel_min": 1.0},
        {"accel_max": -1.0},
        {"tau_range": (0.0, 0.3)},
        {"tau_range": (0.5, 0.3)},
        {"length_range": (-1.0, 4.0)},
        {"vehicle_seed": -1},
        {"init_spacing_noise": -0.1},
        {"spacing_weight": -1.0},
        {"jerk_weight": float("nan")},
        {"collision_penalty": -5.0},
        {"breakup_distance": 10.0},
        {"render_mode": "ansi"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        PlatoonEnv(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_followers": 8.0}, {"dt": "0.1"}, {"tau_range": 0.3}, {"randomize_vehicles": 1}],
)
def test_wrong_argument_types_raise_type_error(kwargs):
    with pytest.raises(TypeError):
        PlatoonEnv(**kwargs)


def test_reset_needed_and_action_validation():
    env = PlatoonEnv(render_mode="rgb_array")
    with pytest.raises(ResetNeededError):
        env.step(np.zeros(N))
    with pytest.raises(ResetNeededError):
        env.render()
    with pytest.raises(ResetNeededError):
        _ = env.positions
    with pytest.raises(ResetNeededError):
        PlatoonVectorEnv(2).step(np.zeros((2, N)))
    env.reset(seed=0)
    with pytest.raises(ValueError, match="NaN"):
        env.step(np.full(N, np.nan))
    with pytest.raises(ValueError):
        env.step(np.array([0.0, np.inf] * (N // 2)))
    with pytest.raises(ValueError):
        env.step(np.zeros(N + 1))
    env.step(np.zeros((1, N)))  # reshapeable actions are accepted
    _, _, _, _, info = env.step(np.full(N, 100.0))
    np.testing.assert_array_equal(info["commands"], 3.0)  # clipped
    with pytest.raises(ValueError, match="together"):
        env.reset(options={"positions": np.zeros(N + 1)})
    with pytest.raises(ValueError, match="behind"):
        env.reset(options={"positions": np.zeros(N + 1), "velocities": np.zeros(N + 1)})
    with pytest.raises(ValueError):
        env.reset(options={"leader_command": np.zeros(3)})
    env.close()


def test_unknown_reset_options_warn_and_are_ignored():
    # PettingZoo's parallel_api_test resets with options={"options": 1}.
    env = PlatoonEnv()
    reference, _ = PlatoonEnv().reset(seed=3)
    with pytest.warns(UserWarning, match="ignoring unknown reset option") as record:
        obs, _ = env.reset(seed=3, options={"options": 1})
    assert record[0].filename == __file__  # reported at the caller of reset()
    np.testing.assert_array_equal(obs, reference)
    with pytest.warns(UserWarning, match="speed"):
        env.reset(seed=0, options={"speed": 1.0, "leader_command": np.zeros(600)})
    assert np.all(env.leader_command == 0.0)  # known keys still apply
    with pytest.warns(UserWarning), pytest.raises(ValueError):
        env.reset(options={"options": 1, "leader_command": np.zeros(3)})  # still strict
    envs = PlatoonVectorEnv(2)
    with pytest.warns(UserWarning, match="ignoring unknown reset option") as record:
        obs, _ = envs.reset(seed=0, options={"options": 1})
    assert record[0].filename == __file__
    assert obs.shape == (2, N, 13)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env.reset(seed=0, options={})
        envs.reset(seed=0, options={"reset_mask": np.array([True, True])})


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_rgb_array_frames(theme):
    previous = rendering.get_theme()
    rendering.set_theme(theme)
    try:
        env = PlatoonEnv(topology="predecessor_leader", render_mode="rgb_array")
        obs, _ = env.reset(seed=0)
        frames = [env.render()]
        for _ in range(8):
            obs, *_ = env.step(cacc_policy(obs))
            frames.append(env.render())
        env.close()
    finally:
        rendering.set_theme(previous)
    for frame in frames:
        assert frame.shape == (560, 1000, 3) and frame.dtype == np.uint8
        assert frame.std() > 5.0
    assert not np.array_equal(frames[0], frames[-1])
    expected = [int(rendering.get_theme(theme).background[k : k + 2], 16) for k in (1, 3, 5)]
    np.testing.assert_allclose(frames[0][2, 2], expected, atol=2)


def test_blitted_frame_matches_full_redraw(light_theme):
    env = PlatoonEnv(scenario="stop_and_go", render_mode="rgb_array", topology="bidirectional")
    obs, _ = env.reset(seed=0)
    for _ in range(40):
        obs, *_ = env.step(acc_policy(obs))
        frame = env.render()
    renderer = env._renderer
    for artist in renderer._dynamic_artists:
        artist.set_animated(False)
    full = figure_to_rgb(renderer.fig)
    assert np.abs(frame.astype(int) - full.astype(int)).mean() < 0.5
    env.close()


def test_render_history_mode_and_close():
    env = PlatoonEnv(render_mode="rgb_array", max_steps=20)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(cacc_policy(obs))
        env.render()
    env.reset(seed=1)
    env.render()
    assert np.count_nonzero(~np.isnan(env._renderer._err_hist[:, 0])) == 1
    env.close()
    env.close()
    assert env.render().shape == (560, 1000, 3)  # re-creates the figure
    env.close()

    quiet = PlatoonEnv()
    quiet.reset(seed=0)
    with pytest.warns(UserWarning, match="render_mode"):
        assert quiet.render() is None

    human = PlatoonEnv(render_mode="human", max_steps=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # non-interactive Agg backend
        human.reset(seed=0)
        human.step(np.zeros(N))
        assert human.render() is None
    human.close()

    envs = PlatoonVectorEnv(2, n_followers=3, render_mode="rgb_array")
    envs.reset(seed=0)
    assert envs.render().shape == (560, 1000, 3)
    envs.close()
