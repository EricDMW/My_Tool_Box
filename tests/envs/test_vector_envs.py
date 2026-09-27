"""Tests for the native batched vector environments.

* :class:`env_lib.consensus_env.vector.ConsensusVectorEnv`
* :class:`env_lib.kos_env.vector.KuramotoOscillatorVectorEnv`

Both share their array kernels with the single environments, so every copy
must reproduce a single environment started from the same state exactly.
"""

from __future__ import annotations

import time
import warnings

import gymnasium as gym
import numpy as np
import pytest

import env_lib
from env_lib.consensus_env import ConsensusEnv, ConsensusVectorEnv, proximity_adjacency
from env_lib.errors import ResetNeededError
from env_lib.kos_env import KuramotoOscillatorEnv, KuramotoOscillatorVectorEnv

CONSENSUS_CONFIGS = [
    {},
    {"task": "formation", "formation_shape": "wedge"},
    {"task": "formation", "topology": "proximity", "dynamics": "double", "comm_radius": 3.0},
    {"topology": "star", "dynamics": "double", "max_neighbors": 3, "control_cost": 0.5},
    {"topology": "complete", "n_agents": 6, "max_neighbors": 2, "task": "formation"},
    {"topology": "erdos_renyi", "graph_seed": 5, "max_neighbors": 7, "n_agents": 10},
    {"topology": "line", "n_agents": 5, "max_neighbors": 6, "arena_size": 3.0},
]
KURAMOTO_CONFIGS = [
    {},
    {"coupling_mode": "constant", "integration_method": "rk4", "normalize_coupling": True},
    {"reward_type": "frequency_synchronization", "n_agents": 3, "topology": "ring"},
    {"reward_type": "combined", "topology": "star", "n_oscillators": 7, "sync_threshold": 0.6},
    {"coupling_mode": "constant", "topology": "random", "reward_type": "phase_coherence"},
]


def consensus_pair(num_envs, **kwargs):
    return ConsensusVectorEnv(num_envs, **kwargs), ConsensusEnv(**kwargs)


def kuramoto_pair(num_envs, **kwargs):
    return KuramotoOscillatorVectorEnv(num_envs, **kwargs), KuramotoOscillatorEnv(**kwargs)


def random_actions(space, rng, scale=1.5):
    low, high = space.low * scale, space.high * scale
    return rng.uniform(low, high).astype(np.float32)


def consensus_state(envs, rng):
    shape = (envs.num_envs, envs.n_agents, 2)
    limit = 0.8 * envs.arena_size
    return {"positions": rng.uniform(-limit, limit, shape), "velocities": rng.normal(size=shape)}


def kuramoto_state(envs, rng):
    batch, n = envs.num_envs, envs.n_oscillators
    state = {
        "phases": rng.uniform(-4.0, 4.0, (batch, n)),
        "natural_frequencies": rng.uniform(0.5, 2.0, (batch, n)),
    }
    if envs.coupling_mode == "dynamic":
        state["coupling_strengths"] = rng.uniform(0.0, 5.0, (batch, envs.n_couplings))
    return state


# ---------------------------------------------------------------------------
# Spaces, shapes and infos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", CONSENSUS_CONFIGS)
def test_consensus_spaces_and_shapes(kwargs):
    envs, env = consensus_pair(4, **kwargs)
    if kwargs.get("topology") != "erdos_renyi" or "max_neighbors" in kwargs:
        # Per-copy random graphs may need more neighbour slots than copy 0 alone.
        assert envs.single_observation_space == env.observation_space
        assert envs.observation_layout == env.observation_layout
    assert envs.single_action_space == env.action_space
    n = env.n_agents
    assert envs.observation_space.shape == (4, n, envs.obs_dim)
    assert envs.action_space.shape == (4, n, 2)
    obs, infos = envs.reset(seed=0)
    assert obs.dtype == np.float32 and envs.observation_space.contains(obs)
    assert infos["agent_rewards"].shape == (4, n) and infos["_agent_rewards"].all()
    obs, rewards, terminated, truncated, infos = envs.step(envs.action_space.sample())
    assert envs.observation_space.contains(obs)
    assert rewards.shape == (4,) and rewards.dtype == np.float64
    assert terminated.dtype == bool and truncated.dtype == bool and terminated.shape == (4,)
    assert set(infos) >= {"agent_rewards", "error", "algebraic_connectivity", "adjacency"}
    assert infos["adjacency"].shape == (4, n, n) and infos["adjacency"].dtype == bool
    assert infos["success"].shape == infos["step"].shape == (4,)
    for key in ("agent_rewards", "error", "algebraic_connectivity", "adjacency", "success", "step"):
        assert infos[f"_{key}"].all()
    np.testing.assert_array_equal(infos["step"], 1)


@pytest.mark.parametrize("kwargs", KURAMOTO_CONFIGS)
def test_kuramoto_spaces_and_shapes(kwargs):
    envs, env = kuramoto_pair(5, **kwargs)
    assert envs.single_observation_space == env.observation_space
    assert envs.single_action_space == env.action_space
    assert envs.observation_space.shape == (5, *env.observation_space.shape)
    assert envs.action_space.shape == (5, *env.action_space.shape)
    obs, infos = envs.reset(seed=0)
    assert obs.dtype == np.float32 and envs.observation_space.contains(obs)
    n = env.n_oscillators
    assert infos["phases"].shape == (5, n) and infos["coupling_matrix"].shape == (5, n, n)
    obs, rewards, terminated, truncated, infos = envs.step(envs.action_space.sample())
    assert envs.observation_space.contains(obs)
    assert rewards.shape == (5,) and terminated.shape == truncated.shape == (5,)
    expected = {
        "order_parameter",
        "phase_coherence",
        "step_count",
        "natural_frequencies",
        "phases",
        "coupling_matrix",
        "device",
        "n_agents",
        "dphases_dt",
        "agent_rewards",
    }
    assert expected <= set(infos)
    assert infos["agent_rewards"].shape == (5, env.n_agents)
    np.testing.assert_allclose(infos["agent_rewards"], np.repeat(rewards[:, None], env.n_agents, 1))
    assert infos["dphases_dt"].shape == (5, n) and infos["coupling_matrix"].shape == (5, n, n)
    assert list(infos["device"]) == ["cpu"] * 5
    np.testing.assert_array_equal(infos["step_count"], 1)


# ---------------------------------------------------------------------------
# Equivalence with the single environments
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", CONSENSUS_CONFIGS)
def test_consensus_copies_match_single_env(kwargs):
    """Every copy reproduces a single environment started from the same state, bit for bit."""
    envs, env = consensus_pair(3, max_steps=25, **kwargs)
    rng = np.random.default_rng(1)
    state = consensus_state(envs, rng)
    obs, infos = envs.reset(seed=0, options=state)
    singles = []
    for b in range(envs.num_envs):
        single = ConsensusEnv(max_steps=25, **kwargs)
        if single.topology == "erdos_renyi" and b > 0:
            continue  # other copies have their own random graph
        single_obs, _ = single.reset(options={key: value[b] for key, value in state.items()})
        np.testing.assert_array_equal(single_obs, obs[b])
        singles.append((b, single))
    for _ in range(25):
        actions = random_actions(envs.action_space, rng)
        if rng.random() < 0.5:
            actions = envs.laplacian_policy()
            for b, single in singles:
                np.testing.assert_array_equal(actions[b], single.laplacian_policy())
        obs, rewards, terminated, truncated, infos = envs.step(actions)
        for b, single in singles:
            s_obs, s_reward, s_term, s_trunc, s_info = single.step(actions[b])
            np.testing.assert_array_equal(obs[b], s_obs)
            assert rewards[b] == s_reward
            assert (terminated[b], truncated[b]) == (s_term, s_trunc)
            np.testing.assert_array_equal(infos["agent_rewards"][b], s_info["agent_rewards"])
            np.testing.assert_array_equal(infos["adjacency"][b], s_info["adjacency"])
            assert infos["error"][b] == s_info["error"]
            assert infos["algebraic_connectivity"][b] == s_info["algebraic_connectivity"]
            assert infos["success"][b] == s_info["success"]
            assert infos["step"][b] == s_info["step"]
        if terminated[0] or truncated[0]:
            break
    np.testing.assert_array_equal(envs.positions[0], singles[0][1].positions)


@pytest.mark.parametrize("kwargs", KURAMOTO_CONFIGS)
def test_kuramoto_copies_match_single_env(kwargs):
    envs, _ = kuramoto_pair(3, max_steps=40, **kwargs)
    rng = np.random.default_rng(2)
    state = kuramoto_state(envs, rng)
    obs, _ = envs.reset(seed=0, options=state)
    singles = []
    for b in range(envs.num_envs):
        single = KuramotoOscillatorEnv(max_steps=40, **kwargs)
        single_obs, _ = single.reset(options={key: value[b] for key, value in state.items()})
        np.testing.assert_array_equal(single_obs, obs[b])
        singles.append(single)
    for _ in range(40):
        actions = random_actions(envs.action_space, rng)
        obs, rewards, terminated, truncated, infos = envs.step(actions)
        for b, single in enumerate(singles):
            s_obs, s_reward, s_term, s_trunc, s_info = single.step(actions[b])
            np.testing.assert_array_equal(obs[b], s_obs)
            assert rewards[b] == s_reward
            assert (terminated[b], truncated[b]) == (s_term, s_trunc)
            for key in ("phases", "dphases_dt", "coupling_matrix", "agent_rewards"):
                np.testing.assert_array_equal(infos[key][b], s_info[key])
            assert infos["order_parameter"][b] == s_info["order_parameter"]
            assert infos["phase_coherence"][b] == s_info["phase_coherence"]
        if terminated.any() or truncated.any():
            break


def test_seeded_resets_are_reproducible():
    for cls, kwargs in [
        (ConsensusVectorEnv, {"topology": "proximity", "noise_std": 0.2}),
        (KuramotoOscillatorVectorEnv, {"noise_std": 0.05}),
    ]:
        a, b = cls(8, **kwargs), cls(8, **kwargs)
        obs_a, _ = a.reset(seed=3)
        obs_b, _ = b.reset(seed=3)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert not np.array_equal(obs_a[0], obs_a[1])  # copies differ
        actions = a.action_space.sample()
        np.testing.assert_array_equal(a.step(actions)[0], b.step(actions)[0])
        assert not np.array_equal(a.reset(seed=4)[0], obs_a)


# ---------------------------------------------------------------------------
# Graph handling (consensus)
# ---------------------------------------------------------------------------
def test_consensus_erdos_renyi_graphs_are_sampled_per_copy():
    envs = ConsensusVectorEnv(6, topology="erdos_renyi", graph_seed=11)
    graphs = envs.adjacency
    assert graphs.shape == (6, 8, 8)
    assert len({graph.tobytes() for graph in graphs}) > 1
    np.testing.assert_array_equal(
        graphs[0], ConsensusEnv(topology="erdos_renyi", graph_seed=11).adjacency
    )
    np.testing.assert_array_equal(graphs, np.swapaxes(graphs, 1, 2))
    # Default slots: maximum degree over all copies.
    assert envs.max_neighbors == graphs.sum(axis=-1).max()
    lambda2 = envs.algebraic_connectivity
    assert lambda2.shape == (6,) and np.all(lambda2 > 1e-9)
    np.testing.assert_array_equal(
        ConsensusVectorEnv(6, topology="erdos_renyi", graph_seed=11).adjacency, graphs
    )


def test_consensus_proximity_graphs_follow_positions():
    envs = ConsensusVectorEnv(5, topology="proximity", comm_radius=3.0)
    obs, infos = envs.reset(seed=0)
    for _ in range(8):
        *_, infos = envs.step(envs.action_space.sample())
        for b in range(5):
            np.testing.assert_array_equal(
                infos["adjacency"][b], proximity_adjacency(envs.positions[b], 3.0)
            )
            laplacian = envs.laplacian[b]
            assert infos["algebraic_connectivity"][b] == pytest.approx(
                max(np.linalg.eigvalsh(laplacian)[1], 0.0), abs=1e-9
            )


def test_consensus_proximity_initial_graphs_are_connected():
    envs = ConsensusVectorEnv(32, topology="proximity")
    _, infos = envs.reset(seed=0)
    assert np.all(infos["algebraic_connectivity"] > 1e-9)
    envs = ConsensusVectorEnv(3, topology="proximity", n_agents=12, comm_radius=0.6)
    with pytest.warns(RuntimeWarning, match="sequential placement"):
        _, infos = envs.reset(seed=0)
    assert np.all(infos["algebraic_connectivity"] > 1e-9)


# ---------------------------------------------------------------------------
# Autoreset and partial resets
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [ConsensusVectorEnv, KuramotoOscillatorVectorEnv])
def test_next_step_autoreset(factory):
    kwargs = {"max_steps": 3}
    if factory is KuramotoOscillatorVectorEnv:
        kwargs["sync_threshold"] = 2.0  # never terminates
    envs = factory(4, **kwargs)
    envs.reset(seed=0)
    zero = np.zeros(envs.action_space.shape, dtype=np.float32)
    for step in range(1, 4):
        obs, rewards, terminated, truncated, _ = envs.step(zero)
        assert truncated.all() == (step == 3) and not terminated.any()
    obs, rewards, terminated, truncated, infos = envs.step(zero)
    np.testing.assert_array_equal(rewards, 0.0)
    assert not truncated.any()
    np.testing.assert_array_equal(envs.step_count, 0)
    assert envs.observation_space.contains(obs)
    # As in SyncVectorEnv, reset copies report their reset info.
    key = "step" if factory is ConsensusVectorEnv else "step_count"
    np.testing.assert_array_equal(infos[key], 0)
    assert infos[f"_{key}"].all()
    if factory is KuramotoOscillatorVectorEnv:
        assert not infos["_dphases_dt"].any()  # step-only key, masked out
        np.testing.assert_allclose(
            infos["order_parameter"], np.abs(np.exp(1j * envs.phases).mean(axis=1))
        )
    envs.step(zero)
    np.testing.assert_array_equal(envs.step_count, 1)


def test_next_step_autoreset_mixes_step_and_reset_infos():
    envs = KuramotoOscillatorVectorEnv(2, n_oscillators=4, max_steps=5, sync_threshold=2.0)
    envs.reset(seed=0)
    zero = np.zeros(envs.action_space.shape, dtype=np.float32)
    envs.step(zero)
    # Only copy 1 is reset (partial reset with the "disabled"-style mask is not
    # available in next_step mode, so shorten copy 1's episode instead).
    envs._step[1] = envs.max_steps - 1
    *_, truncated, _ = envs.step(zero)
    assert truncated.tolist() == [False, True]
    *_, infos = envs.step(zero)
    assert infos["_dphases_dt"].tolist() == [True, False]
    np.testing.assert_array_equal(infos["step_count"], [3, 0])
    assert infos["_step_count"].all()


@pytest.mark.parametrize("factory", [ConsensusVectorEnv, KuramotoOscillatorVectorEnv])
def test_same_step_autoreset_keeps_final_observation(factory):
    envs = factory(3, max_steps=2, autoreset_mode="same_step")
    envs.reset(seed=0)
    zero = np.zeros(envs.action_space.shape, dtype=np.float32)
    first, *_ = envs.step(zero)
    obs, rewards, terminated, truncated, infos = envs.step(zero)
    assert truncated.all()
    assert infos["_final_obs"].all() and infos["_final_info"].all()
    for b in range(3):
        assert infos["final_obs"][b].shape == envs.single_observation_space.shape
    final_info = infos["final_info"]  # batched dict, as in gymnasium's SyncVectorEnv
    key = "step" if factory is ConsensusVectorEnv else "step_count"
    np.testing.assert_array_equal(final_info[key], 2)
    assert final_info[f"_{key}"].all()
    np.testing.assert_array_equal(envs.step_count, 0)
    assert not np.array_equal(obs, infos["final_obs"][0][None])


def test_disabled_autoreset_and_partial_reset_with_state():
    envs = ConsensusVectorEnv(3, autoreset_mode="disabled", max_steps=1)
    envs.reset(seed=0)
    envs.step(envs.laplacian_policy())
    before = envs.positions
    start = np.zeros((3, 8, 2))
    start[2] = 1.5
    mask = np.array([False, False, True])
    obs, infos = envs.reset(options={"reset_mask": mask, "positions": start})
    after = envs.positions
    np.testing.assert_array_equal(after[:2], before[:2])
    np.testing.assert_array_equal(after[2], 1.5)
    np.testing.assert_array_equal(envs.step_count, [1, 1, 0])
    np.testing.assert_array_equal(infos["_error"], mask)
    np.testing.assert_array_equal(obs[2, :, 0:2], 1.5)

    kos = KuramotoOscillatorVectorEnv(2, autoreset_mode="disabled", n_oscillators=4)
    kos.reset(seed=0)
    kos.step(kos.action_space.sample())
    kos.reset(options={"reset_mask": np.array([True, False]), "phases": np.full(4, 0.5)})
    np.testing.assert_array_equal(kos.phases[0], 0.5)
    np.testing.assert_array_equal(kos.step_count, [0, 1])


def test_reset_option_broadcasting_and_validation():
    envs = ConsensusVectorEnv(4, dynamics="double")
    obs, _ = envs.reset(options={"positions": np.ones((8, 2)), "velocities": np.zeros((8, 2))})
    np.testing.assert_array_equal(envs.positions, 1.0)
    with pytest.raises(ValueError):
        envs.reset(options={"positions": np.zeros((3, 8, 2))})
    with pytest.warns(UserWarning, match="ignoring unknown reset option") as record:
        envs.reset(options={"speed": 1.0, "positions": np.full((8, 2), 2.0)})
    assert record[0].filename == __file__
    np.testing.assert_array_equal(envs.positions, 2.0)
    kos = KuramotoOscillatorVectorEnv(4, n_oscillators=3)
    with pytest.raises(ValueError, match="rows"):
        kos.reset(options={"phases": np.zeros((2, 3))})
    with pytest.warns(UserWarning, match="ignoring unknown reset option") as record:
        kos.reset(options={"options": 1})  # as passed by PettingZoo's API test
    assert record[0].filename == __file__  # the warning points at the caller


def test_errors_before_reset_and_bad_actions():
    for envs in (ConsensusVectorEnv(2), KuramotoOscillatorVectorEnv(2)):
        with pytest.raises(ResetNeededError):
            envs.step(envs.action_space.sample())
        envs.reset(seed=0)
        with pytest.raises(ValueError):
            envs.step(np.zeros((3, 5)))
        bad = envs.action_space.sample()
        bad.flat[0] = np.nan
        with pytest.raises(ValueError):
            envs.step(bad)


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (ConsensusVectorEnv, {"n_agents": 1}),
        (ConsensusVectorEnv, {"topology": "grid"}),
        (ConsensusVectorEnv, {"dt": 0.0}),
        (ConsensusVectorEnv, {"num_envs": 0}),
        (KuramotoOscillatorVectorEnv, {"n_oscillators": 1}),
        (KuramotoOscillatorVectorEnv, {"coupling_mode": "adaptive"}),
        (KuramotoOscillatorVectorEnv, {"autoreset_mode": "sometimes"}),
    ],
)
def test_invalid_arguments(factory, kwargs):
    kwargs = {"num_envs": 2, **kwargs}
    with pytest.raises(ValueError):
        factory(**kwargs)


def test_actions_are_clipped():
    envs = ConsensusVectorEnv(2, n_agents=2, arena_size=5.0, max_control=0.5)
    envs.reset(options={"positions": np.zeros((2, 2))})
    envs.step(np.full((2, 2, 2), 10.0, dtype=np.float32))
    np.testing.assert_allclose(envs.velocities, 0.5)
    kos = KuramotoOscillatorVectorEnv(2, n_oscillators=3, coupling_mode="constant")
    kos.reset(seed=0)
    kos.step(np.full((2, 3), 7.0, dtype=np.float32))
    np.testing.assert_array_equal(kos.control_inputs, 1.0)


def test_consensus_laplacian_policy_solves_batched_task():
    envs = ConsensusVectorEnv(16, task="formation", autoreset_mode="disabled")
    envs.reset(seed=0)
    done = np.zeros(16, dtype=bool)
    for _ in range(envs.max_steps):
        _, _, terminated, truncated, infos = envs.step(envs.laplacian_policy())
        done |= terminated
        if done.all():
            break
    assert done.all() and infos["success"].all()


# ---------------------------------------------------------------------------
# Registration, make_vec and throughput
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:.*is out of date:DeprecationWarning")
@pytest.mark.parametrize(
    "env_id, cls",
    [
        ("Consensus-v0", ConsensusVectorEnv),
        ("Formation-v0", ConsensusVectorEnv),
        ("KuramotoOscillator-v0", KuramotoOscillatorVectorEnv),
        ("KuramotoOscillator-Constant-v0", KuramotoOscillatorVectorEnv),
        ("KuramotoOscillator-FreqSync-Constant-v0", KuramotoOscillatorVectorEnv),
    ],
)
def test_make_vec_returns_native_class(env_id, cls):
    envs = env_lib.make_vec(env_id, 4)
    assert type(envs) is cls
    single = env_lib.make(env_id).unwrapped
    assert envs.single_observation_space == single.observation_space
    assert envs.metadata["autoreset_mode"] is not None
    obs, _ = envs.reset(seed=0)
    obs, *_ = envs.step(envs.action_space.sample())
    assert envs.observation_space.contains(obs)
    envs.close()
    if env_id == "Formation-v0":
        assert envs.task == "formation"
    assert env_lib.ConsensusVectorEnv is ConsensusVectorEnv
    assert env_lib.KuramotoOscillatorVectorEnv is KuramotoOscillatorVectorEnv


@pytest.mark.filterwarnings("ignore:.*is out of date:DeprecationWarning")
@pytest.mark.parametrize("env_id", ["Formation-v0", "KuramotoOscillator-v0"])
def test_sync_vectorization_still_works(env_id):
    envs = env_lib.make_vec(env_id, 3, vectorization_mode="sync")
    assert isinstance(envs, gym.vector.SyncVectorEnv)
    obs, _ = envs.reset(seed=0)
    obs, rewards, *_ = envs.step(envs.action_space.sample())
    assert obs.shape == (3, *envs.single_observation_space.shape) and rewards.shape == (3,)
    envs.close()


def test_make_vec_forwards_kwargs():
    envs = env_lib.make_vec("Consensus-v0", 2, n_agents=5, autoreset_mode="same_step")
    assert envs.n_agents == 5 and envs.autoreset_mode == "same_step"
    envs = env_lib.make_vec("KuramotoOscillator-v1", 2, integration_method="rk4")
    assert envs.n_oscillators == 5 and envs.integration_method == "rk4"


def _steps_per_second(envs, steps):
    envs.reset(seed=0)
    actions = envs.action_space.sample()
    envs.step(actions)
    start = time.perf_counter()
    for _ in range(steps):
        envs.step(actions)
    return steps / (time.perf_counter() - start)


@pytest.mark.filterwarnings("ignore:.*is out of date:DeprecationWarning")
@pytest.mark.parametrize("env_id", ["Consensus-v0", "KuramotoOscillator-v0"])
def test_native_is_faster_than_sync(env_id):
    """Loose sanity check: the batched implementation beats SyncVectorEnv at 64 copies."""
    native = _steps_per_second(env_lib.make_vec(env_id, 64), 30)
    sync = _steps_per_second(env_lib.make_vec(env_id, 64, vectorization_mode="sync"), 5)
    assert native > 2.0 * sync


# ---------------------------------------------------------------------------
# Rendering (copy 0)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [ConsensusVectorEnv, KuramotoOscillatorVectorEnv])
def test_render_copy_zero(factory):
    envs = factory(3, render_mode="rgb_array")
    envs.reset(seed=0)
    first = envs.render()
    envs.step(envs.action_space.sample())
    frame = envs.render()
    assert frame.shape == (560, 1000, 3) and frame.dtype == np.uint8
    assert not np.array_equal(first, frame)
    envs.close()
    envs.close()
    assert factory(2).render() is None


def test_human_render_mode_is_headless_safe():
    envs = ConsensusVectorEnv(2, render_mode="human", max_steps=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # non-interactive Agg backend
        envs.reset(seed=0)
        envs.step(envs.action_space.sample())
    envs.close()
