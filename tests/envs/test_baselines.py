"""Tests for env_lib.baselines: every baseline beats random actions, batching, dispatch."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import env_lib
from env_lib.baselines import (
    BaselinePolicy,
    baseline_policy,
    family_of,
    kuramoto_feedback,
    laplacian_consensus,
    list_baselines,
)
from env_lib.registration import ENV_SPECS
from env_lib.utils import evaluate
from env_lib.wrappers import FlattenJointSpaces, TeamReward, to_parallel


def _compare(env_id, n_episodes=3, **kwargs):
    env = env_lib.make(env_id, **kwargs)
    random = evaluate(env, None, n_episodes=n_episodes, seed=0)
    baseline = evaluate(env, baseline_policy(env), n_episodes=n_episodes, seed=0)
    env.close()
    return random, baseline


# Environments whose return grows with task performance: the baseline must win on return.
RETURN_CASES = [
    ("Consensus-v0", {}),
    ("Formation-v0", {"dynamics": "double", "formation_shape": "wedge"}),
    ("LineMsg-v0", {}),
    ("LineMsg-v0", {"action_space_type": "multibinary", "num_agents": 6}),
    ("WirelessComm-v1", {}),
    ("WirelessComm-v0", {"max_iter": 20}),
    # Kuramoto with synchronisation termination disabled (see the next test).
    ("KuramotoOscillator-v0", {"sync_threshold": 1.5, "max_steps": 100}),
    ("KuramotoOscillator-Constant-v0", {"sync_threshold": 1.5, "max_steps": 150}),
    ("KuramotoOscillator-FreqSync-Constant-v0", {}),
    # AJLATT with collision termination disabled (see test_ajlatt_baseline_avoids_collisions).
    ("AJLATT-v0", {"terminate_on_collision": False, "max_episode_steps": 25}),
]


@pytest.mark.parametrize(("env_id", "kwargs"), RETURN_CASES)
def test_baseline_beats_random_return(env_id, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        random, baseline = _compare(env_id, n_episodes=2 if "AJLATT" in env_id else 3, **kwargs)
    assert baseline.mean_return > random.mean_return


def test_pistonball_baseline_beats_random():
    pytest.importorskip("pymunk")
    random, baseline = _compare("Pistonball-v0", n_pistons=10)
    assert baseline.mean_return > random.mean_return
    assert baseline.termination_rate > random.termination_rate  # the ball reaches the goal
    random, baseline = _compare("Pistonball-v0", n_pistons=10, continuous=False)
    assert baseline.mean_return > random.mean_return


@pytest.mark.parametrize("env_id", ["KuramotoOscillator-v1", "KuramotoOscillator-Constant-v0"])
def test_kuramoto_baseline_synchronises_faster(env_id):
    # With the default termination on synchronisation (and a positive per-step
    # reward) the return rewards slow synchronisation, so the task metric is
    # the time to synchronise.
    random, baseline = _compare(env_id, n_episodes=4)
    assert baseline.termination_rate >= random.termination_rate
    assert baseline.mean_length < random.mean_length


def test_kuramoto_torch_baseline():
    pytest.importorskip("torch")
    random, baseline = _compare(
        "KuramotoOscillatorTorch-Constant-v0", n_episodes=2, sync_threshold=1.5, max_steps=100
    )
    assert baseline.mean_return > random.mean_return


def test_ajlatt_baseline_avoids_collisions():
    random, baseline = _compare("AJLATT-v0", n_episodes=2, max_episode_steps=25)
    assert random.termination_rate == 1.0  # random robots collide
    assert baseline.termination_rate == 0.0


@pytest.mark.parametrize("env_id", ["PowerGrid-v0", "Platoon-v0"])
def test_new_environment_baselines_beat_random(env_id):
    package = {"PowerGrid-v0": "env_lib.power_grid_env", "Platoon-v0": "env_lib.platoon_env"}
    pytest.importorskip(package[env_id])
    try:
        random, baseline = _compare(env_id, n_episodes=2)
    except NotImplementedError as exc:  # environment still in development
        pytest.skip(str(exc))
    assert baseline.mean_return > random.mean_return


def test_list_baselines_covers_every_family():
    families = {spec.family for spec in ENV_SPECS}
    assert families <= set(list_baselines())
    assert all(isinstance(text, str) and text.isascii() for text in list_baselines().values())


def test_consensus_baseline_equals_laplacian_policy():
    for kwargs in ({}, {"task": "formation"}, {"dynamics": "double"}, {"topology": "star"}):
        env = env_lib.make("Consensus-v0", **kwargs)
        policy = baseline_policy(env)
        obs, _ = env.reset(seed=1)
        for _ in range(15):
            action = policy(obs)
            np.testing.assert_allclose(action, env.unwrapped.laplacian_policy(), atol=1e-5)
            obs, *_ = env.step(action)


def _reference_encircle(core, radius=1.8, gain=1.2):
    """The heuristic of the former examples/ajlatt_demo.py (reads the beliefs directly)."""
    actions = np.zeros((core.num_robots, 2))
    v_max = core.config.max_linear_velocity
    w_max = core.config.max_angular_velocity
    for i in range(core.num_robots):
        x, y, theta = core.robot_est[i].state
        target = core.target_est[i][0].state[:2]
        angle = 2 * np.pi * i / core.num_robots
        goal = target + radius * np.array([np.cos(angle), np.sin(angle)])
        delta = goal - np.array([x, y])
        heading = np.arctan2(delta[1], delta[0])
        if np.linalg.norm(delta) < 0.3:
            heading = np.arctan2(target[1] - y, target[0] - x)
        error = (heading - theta + np.pi) % (2 * np.pi) - np.pi
        speed = min(v_max, 0.5 * np.linalg.norm(delta)) * max(0.0, np.cos(error))
        actions[i] = (speed, np.clip(gain * error, -w_max, w_max))
    return actions


def test_ajlatt_baseline_matches_belief_based_heuristic():
    env = env_lib.make("AJLATT-v0", max_episode_steps=10)
    policy = baseline_policy(env)
    obs, _ = env.reset(seed=0)
    for _ in range(8):
        action = policy(obs)
        np.testing.assert_allclose(action, _reference_encircle(env.unwrapped), atol=1e-4)
        obs, *_ = env.step(action)


BATCH_IDS = [
    "Consensus-v0",
    "KuramotoOscillator-v0",
    "KuramotoOscillator-Constant-v0",
    "LineMsg-v0",
    "WirelessComm-v1",
    "AJLATT-v0",
]


@pytest.mark.parametrize("env_id", BATCH_IDS)
def test_policy_accepts_any_leading_batch_shape(env_id):
    env = env_lib.make(env_id)
    policy = baseline_policy(env)
    first, _ = env.reset(seed=0)
    second, *_ = env.step(policy(first))
    single = [np.asarray(policy(first)), np.asarray(policy(second))]
    batch = np.asarray(policy(np.stack([first, second])))
    np.testing.assert_allclose(batch, np.stack(single), atol=1e-6)
    nested = np.asarray(policy(np.stack([first, second])[None]))
    assert nested.shape == (1, *batch.shape)
    assert env.action_space.contains(np.asarray(single[0]).astype(env.action_space.dtype))
    env.close()


def test_vector_env_dispatch_sync_and_native():
    sync = env_lib.make_vec("WirelessComm-v1", num_envs=3, vectorization_mode="sync")
    policy = baseline_policy(sync)
    obs, _ = sync.reset(seed=0)
    assert policy(obs).shape == (3, 16)
    sync.step(policy(obs))
    sync.close()
    for env_id in ("Formation-v0", "KuramotoOscillator-v0"):
        try:
            native = env_lib.make_vec(env_id, num_envs=4, vectorization_mode="vector_entry_point")
        except ImportError:
            pytest.skip("native vector environment not available")
        policy = baseline_policy(native)
        obs, _ = native.reset(seed=0)
        action = policy(obs)
        assert action.shape == native.action_space.shape
        native.step(action)
        native.close()


def test_native_vector_matches_single_env_consensus():
    try:
        envs = env_lib.make_vec("Consensus-v0", num_envs=2, vectorization_mode="vector_entry_point")
    except ImportError:
        pytest.skip("native vector environment not available")
    env = env_lib.make("Consensus-v0")
    obs, _ = envs.reset(seed=0)
    single_obs, _ = env.reset(options={"positions": envs.unwrapped.positions[0]})
    np.testing.assert_allclose(baseline_policy(envs)(obs)[0], baseline_policy(env)(single_obs))
    envs.close()


def test_async_vector_env_dispatch():
    envs = env_lib.make_vec("LineMsg-v0", num_envs=2, vectorization_mode="async", num_agents=4)
    try:
        policy = baseline_policy(envs)
        obs, _ = envs.reset(seed=0)
        np.testing.assert_array_equal(policy(obs), [15, 15])
        assert family_of(envs) == "linemsg"
    finally:
        envs.close()


def test_torch_tensors_round_trip():
    torch = pytest.importorskip("torch")
    env = env_lib.make("KuramotoOscillatorTorch-v1")
    policy = baseline_policy(env)
    env.reset(seed=0)
    batch = env.unwrapped.get_batch_observations()
    action = policy(batch)
    assert isinstance(action, torch.Tensor) and action.shape == (4, env.action_space.shape[0])
    env.step(action)  # one action row per parallel system


def test_wrapped_environments():
    flat = FlattenJointSpaces(env_lib.make("Formation-v0"))
    policy = baseline_policy(flat)
    obs, _ = flat.reset(seed=0)
    action = policy(obs)
    assert action.shape == flat.action_space.shape == (16,)
    reference = baseline_policy(flat.unwrapped)(obs.reshape(8, -1)).reshape(-1)
    np.testing.assert_allclose(action, reference)
    team = TeamReward(env_lib.make("AJLATT-v0", max_episode_steps=5))
    obs, _ = team.reset(seed=0)
    assert baseline_policy(team)(obs).shape == (4, 2)


def test_parallel_adapter_policy_maps_dicts():
    par_env = to_parallel("Consensus-v0")
    policy = baseline_policy(par_env)
    observations, _ = par_env.reset(seed=0)
    actions = policy(observations)
    assert set(actions) == set(par_env.agents)
    joint = np.stack([actions[agent] for agent in par_env.possible_agents])
    np.testing.assert_allclose(joint, par_env.env.unwrapped.laplacian_policy(), atol=1e-5)
    par_env.step(actions)
    line = to_parallel("LineMsg-v0", num_agents=3)
    observations, _ = line.reset(seed=0)
    assert {int(a) for a in baseline_policy(line)(observations).values()} == {1}


def test_string_ids_parameters_and_errors():
    policy = baseline_policy("Formation-v0", gain=0.5)
    assert isinstance(policy, BaselinePolicy)
    assert policy.parameters["gain"] == 0.5 and policy.name == "baseline:consensus"
    assert "laplacian_consensus" in repr(policy)
    with pytest.raises(TypeError, match="unknown parameter"):
        baseline_policy("Consensus-v0", radius=1.0)
    with pytest.raises(ValueError):
        baseline_policy("Consensus-v0", gain=-1.0)
    with pytest.raises(TypeError, match="no baseline controller"):
        baseline_policy(_ForeignEnv())
    assert family_of("AJLATT-v0") == "ajlatt"


class _ForeignEnv:
    """Not an env_lib environment."""

    unwrapped = None
    observation_space = action_space = None


def test_pure_functions_validate_shapes():
    with pytest.raises(ValueError):
        laplacian_consensus(np.zeros((4, 7)))
    action = kuramoto_feedback(np.zeros((2, 18)), n_oscillators=6)
    assert action.shape == (2, 6) and action.dtype == np.float32
