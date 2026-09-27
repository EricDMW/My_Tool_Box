"""Tests for :class:`env_lib.consensus_env.ConsensusEnv`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env

import env_lib
from env_lib.consensus_env import (
    FORMATION_SHAPES,
    ConsensusEnv,
    algebraic_connectivity,
    is_connected,
    make_formation,
    make_topology,
    proximity_adjacency,
)
from env_lib.utils import rendering
from env_lib.utils.rendering import figure_to_rgb

STATIC_TOPOLOGIES = ("ring", "line", "star", "complete", "erdos_renyi")


# ---------------------------------------------------------------------------
# Loop references
# ---------------------------------------------------------------------------
def reference_agent_rewards(pos, offsets, adjacency, control, control_cost):
    """Per-agent reward written agent by agent."""
    n = len(pos)
    rewards = np.zeros(n)
    for i in range(n):
        neighbours = [j for j in range(n) if adjacency[i, j]]
        disagreement = 0.0
        for j in neighbours:
            gap = (pos[i] - offsets[i]) - (pos[j] - offsets[j])
            disagreement += float(gap @ gap)
        if neighbours:
            disagreement /= len(neighbours)
        rewards[i] = -disagreement - control_cost * float(control[i] @ control[i])
    return rewards


def reference_error(pos, offsets):
    shifted = pos - offsets
    centre = shifted.mean(axis=0)
    return float(np.mean([np.sum((y - centre) ** 2) for y in shifted]))


def reference_observation(pos, vel, offsets, adjacency, max_neighbors):
    """Observation rows built agent by agent (neighbours sorted by distance, then index)."""
    n = len(pos)
    rows = []
    for i in range(n):
        row = [*pos[i], *vel[i], *offsets[i]]
        neighbours = [j for j in range(n) if adjacency[i, j]]
        neighbours.sort(key=lambda j: (float(np.sum((pos[j] - pos[i]) ** 2)), j))
        for k in range(max_neighbors):
            if k < len(neighbours):
                j = neighbours[k]
                rel = (pos[j] - offsets[j]) - (pos[i] - offsets[i])
                row += [*rel, *(vel[j] - vel[i]), 1.0]
            else:
                row += [0.0] * 5
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def rollout(env, policy, seed):
    env.reset(seed=seed)
    for _ in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(policy())
        if terminated or truncated:
            return obs, reward, terminated, truncated, info
    raise AssertionError("episode did not end")  # pragma: no cover


@pytest.fixture
def light_theme():
    previous = rendering.get_theme()
    rendering.set_theme("light")
    try:
        yield
    finally:
        rendering.set_theme(previous)


# ---------------------------------------------------------------------------
# Spaces and observations
# ---------------------------------------------------------------------------
def test_default_spaces_and_dtypes():
    env = ConsensusEnv()
    assert env.action_space == Box(-1.0, 1.0, shape=(8, 2), dtype=np.float32)
    assert isinstance(env.observation_space, Box)
    assert env.max_neighbors == 2  # max degree of the ring
    assert env.observation_space.shape == (8, 6 + 5 * 2)
    assert env.observation_space.dtype == np.float32
    assert np.all(np.isinf(env.observation_space.low))
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == (8, 16)
    assert env.observation_space.contains(obs)
    assert set(info) >= {"agent_rewards", "error", "algebraic_connectivity", "adjacency", "success"}
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs) and np.all(np.isfinite(obs))
    assert type(reward) is float
    assert type(terminated) is bool and type(truncated) is bool
    assert info["agent_rewards"].shape == (8,) and info["agent_rewards"].dtype == np.float64
    assert info["adjacency"].dtype == bool and info["adjacency"].shape == (8, 8)
    assert isinstance(info["error"], float) and isinstance(info["success"], bool)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"task": "formation", "dynamics": "double"},
        {"task": "formation", "topology": "proximity", "formation_shape": "wedge"},
        {"topology": "star", "max_neighbors": 3},
        {"topology": "ring", "max_neighbors": 12, "n_agents": 5},
        {"topology": "complete", "n_agents": 6, "max_neighbors": 2, "task": "formation"},
    ],
)
def test_observation_matches_reference(kwargs):
    env = ConsensusEnv(graph_seed=0, **kwargs)
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs, *_rest, info = env.step(rng.uniform(-1, 1, size=(env.n_agents, 2)))
        expected = reference_observation(
            env.positions,
            env.velocities,
            env.formation_offsets,
            info["adjacency"],
            env.max_neighbors,
        )
        np.testing.assert_allclose(obs, expected, rtol=1e-6, atol=1e-5)
        assert env.observation_space.contains(obs)


def test_observation_layout():
    env = ConsensusEnv(task="formation", max_neighbors=3)
    layout = env.observation_layout
    assert layout == {
        "position": slice(0, 2),
        "velocity": slice(2, 4),
        "offset": slice(4, 6),
        "neighbors": slice(6, 21),
    }
    assert env.obs_dim == 21
    obs, _ = env.reset(seed=0)
    np.testing.assert_allclose(obs[:, layout["position"]], env.positions, rtol=1e-6)
    np.testing.assert_allclose(obs[:, layout["offset"]], env.formation_offsets, rtol=1e-6)
    slots = obs[:, layout["neighbors"]].reshape(env.n_agents, env.max_neighbors, env.SLOT_DIM)
    mask = slots[..., env.neighbor_slot_layout["mask"]][..., 0]
    # Ring: two neighbours, the third slot is zero padding.
    np.testing.assert_array_equal(mask, np.tile([1.0, 1.0, 0.0], (env.n_agents, 1)))
    assert np.all(slots[:, 2, :] == 0.0)
    # Slots hold y_j - y_i (y = x - d), sorted by the physical distance ||x_j - x_i||.
    rel_y = slots[:, :2, env.neighbor_slot_layout["rel_position"]].astype(np.float64)
    offsets = env.formation_offsets
    for i in range(env.n_agents):
        neighbours = np.flatnonzero(env.adjacency[i])
        # Match each slot to a neighbour through y_j - y_i, then check the distance order.
        gaps = (env.positions[neighbours] - offsets[neighbours]) - (env.positions[i] - offsets[i])
        order = [int(np.argmin(np.linalg.norm(gaps - slot, axis=1))) for slot in rel_y[i]]
        distances = np.linalg.norm(env.positions[neighbours[order]] - env.positions[i], axis=1)
        assert sorted(order) == [0, 1]
        assert distances[0] <= distances[1]


def test_consensus_offsets_are_zero():
    env = ConsensusEnv()
    obs, _ = env.reset(seed=0)
    assert np.all(obs[:, env.observation_layout["offset"]] == 0.0)
    assert np.all(env.formation_offsets == 0.0)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("topology", ["ring", "proximity"])
def test_seeding_determinism(topology):
    kwargs = dict(topology=topology, noise_std=0.2, dynamics="double", task="formation")
    env_a, env_b = ConsensusEnv(**kwargs), ConsensusEnv(**kwargs)
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)
    rng = np.random.default_rng(1)
    for _ in range(20):
        action = rng.uniform(-1, 1, size=(8, 2))
        out_a, out_b = env_a.step(action), env_b.step(action)
        np.testing.assert_array_equal(out_a[0], out_b[0])
        assert out_a[1] == out_b[1]
    obs_c, _ = env_a.reset(seed=43)
    assert not np.array_equal(obs_a, obs_c)


def test_initial_state_distribution():
    env = ConsensusEnv(arena_size=5.0)
    for seed in range(10):
        env.reset(seed=seed)
        assert np.all(np.abs(env.positions) <= 0.8 * 5.0)
        assert np.all(env.velocities == 0.0)


def test_erdos_renyi_graph_seed_is_reproducible():
    a = ConsensusEnv(topology="erdos_renyi", graph_seed=7)
    b = ConsensusEnv(topology="erdos_renyi", graph_seed=7)
    np.testing.assert_array_equal(a.adjacency, b.adjacency)
    # The graph does not depend on the episode seed.
    a.reset(seed=0)
    first = a.adjacency
    a.reset(seed=1)
    np.testing.assert_array_equal(a.adjacency, first)


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("topology", STATIC_TOPOLOGIES)
@pytest.mark.parametrize("n_agents", [2, 3, 8, 15])
def test_static_topologies_are_connected(topology, n_agents):
    adjacency = make_topology(topology, n_agents, rng=np.random.default_rng(0))
    assert adjacency.dtype == bool and adjacency.shape == (n_agents, n_agents)
    np.testing.assert_array_equal(adjacency, adjacency.T)
    assert not adjacency.diagonal().any()
    assert is_connected(adjacency)
    assert algebraic_connectivity(adjacency) > 1e-9
    env = ConsensusEnv(n_agents=n_agents, topology=topology, graph_seed=0)
    _, info = env.reset(seed=0)
    assert info["algebraic_connectivity"] > 1e-9
    assert env.max_neighbors == adjacency.sum(axis=1).max() or topology == "erdos_renyi"


def test_topology_structure():
    degrees = {
        "ring": [2] * 6,
        "line": [1, 2, 2, 2, 2, 1],
        "star": [5, 1, 1, 1, 1, 1],
        "complete": [5] * 6,
    }
    for topology, expected in degrees.items():
        np.testing.assert_array_equal(make_topology(topology, 6).sum(axis=1), expected)
    # Known Fiedler values.
    assert algebraic_connectivity(make_topology("complete", 6)) == pytest.approx(6.0)
    assert algebraic_connectivity(make_topology("ring", 8)) == pytest.approx(
        2 - 2 * np.cos(2 * np.pi / 8)
    )
    assert algebraic_connectivity(make_topology("star", 6)) == pytest.approx(1.0)
    assert not is_connected(np.zeros((3, 3), dtype=bool))
    assert algebraic_connectivity(np.zeros((3, 3))) == pytest.approx(0.0, abs=1e-12)


def test_erdos_renyi_failure_raises():
    with pytest.raises(ValueError, match="edge_probability"):
        make_topology(
            "erdos_renyi", 40, edge_probability=0.01, rng=np.random.default_rng(0), max_attempts=5
        )


def test_proximity_initial_graph_is_connected():
    env = ConsensusEnv(topology="proximity")
    assert env.max_neighbors == 6
    for seed in range(15):
        _, info = env.reset(seed=seed)
        assert info["algebraic_connectivity"] > 1e-9
        assert is_connected(info["adjacency"])
        np.testing.assert_array_equal(
            info["adjacency"], proximity_adjacency(env.positions, env.comm_radius)
        )


def test_proximity_graph_is_rebuilt_every_step():
    env = ConsensusEnv(topology="proximity", comm_radius=3.0)
    env.reset(seed=0)
    for _ in range(10):
        *_, info = env.step(env.action_space.sample())
        np.testing.assert_array_equal(
            info["adjacency"], proximity_adjacency(env.positions, env.comm_radius)
        )
        assert info["algebraic_connectivity"] == pytest.approx(
            algebraic_connectivity(info["adjacency"]), abs=1e-9
        )


def test_proximity_sequential_fallback_when_radius_is_tiny():
    env = ConsensusEnv(topology="proximity", n_agents=12, comm_radius=0.6)
    with pytest.warns(RuntimeWarning, match="sequential placement"):
        _, info = env.reset(seed=0)
    assert is_connected(info["adjacency"])
    assert np.all(np.abs(env.positions) <= 0.8 * env.arena_size + 1e-12)


# ---------------------------------------------------------------------------
# Rewards, errors and termination
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"task": "formation", "formation_shape": "grid", "dynamics": "double"},
        {"topology": "proximity", "comm_radius": 3.0, "task": "formation", "noise_std": 0.3},
        {"topology": "star", "control_cost": 0.5, "max_control": 2.0},
    ],
)
def test_reward_matches_loop_reference(kwargs):
    env = ConsensusEnv(**kwargs)
    env.reset(seed=11)
    rng = np.random.default_rng(5)
    for _ in range(15):
        action = rng.uniform(-3, 3, size=(env.n_agents, 2))  # partly out of bounds
        _, reward, terminated, _, info = env.step(action)
        clipped = np.clip(action, -env.max_control, env.max_control)
        expected = reference_agent_rewards(
            env.positions, env.formation_offsets, info["adjacency"], clipped, env.control_cost
        )
        np.testing.assert_allclose(info["agent_rewards"], expected, rtol=1e-10, atol=1e-10)
        bonus = env.success_bonus if terminated else 0.0
        assert reward == pytest.approx(expected.mean() + bonus)
        assert info["error"] == pytest.approx(reference_error(env.positions, env.formation_offsets))


def test_isolated_agent_only_pays_control_cost():
    env = ConsensusEnv(topology="proximity", n_agents=3, comm_radius=2.0, control_cost=0.1)
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [6.0, 6.0]])
    env.reset(seed=0, options={"positions": positions})
    action = np.array([[0.0, 0.0], [0.0, 0.0], [0.5, -0.5]])
    _, _, _, _, info = env.step(action)
    assert not info["adjacency"][2].any()
    assert info["agent_rewards"][2] == pytest.approx(-0.1 * 0.5)
    assert info["algebraic_connectivity"] == pytest.approx(0.0, abs=1e-9)


def test_success_bonus_is_paid_once():
    env = ConsensusEnv(task="formation", formation_shape="line")
    start = env.formation_offsets + np.array([1.0, -2.0])
    env.reset(seed=0, options={"positions": start})
    zero = np.zeros((env.n_agents, 2))
    _, reward, terminated, truncated, info = env.step(zero)
    assert terminated and not truncated and info["success"]
    assert reward == pytest.approx(env.success_bonus)
    _, reward, terminated, _, info = env.step(zero)
    assert terminated and info["success"]
    assert reward == pytest.approx(0.0)


def test_truncation():
    env = ConsensusEnv(max_steps=5)
    env.reset(seed=0)
    zero = np.zeros((8, 2))
    for step in range(1, 6):
        _, _, terminated, truncated, info = env.step(zero)
        assert not terminated
        assert truncated == (step == 5)
        assert info["step"] == step


@pytest.mark.parametrize("task", ["consensus", "formation"])
@pytest.mark.parametrize("dynamics", ["single", "double"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_laplacian_policy_solves_default_task(task, dynamics, seed):
    env = ConsensusEnv(task=task, dynamics=dynamics)
    _, reward, terminated, truncated, info = rollout(env, env.laplacian_policy, seed)
    assert terminated and info["success"]
    assert not truncated and info["step"] < env.max_steps
    assert info["error"] < env.tolerance
    assert reward > env.success_bonus - 1.0


@pytest.mark.parametrize("shape", FORMATION_SHAPES)
def test_laplacian_policy_solves_all_formation_shapes(shape):
    env = ConsensusEnv(task="formation", formation_shape=shape)
    *_, terminated, _, info = rollout(env, env.laplacian_policy, 7)
    assert terminated and info["success"]
    shifted = env.positions - env.formation_offsets
    assert np.max(np.linalg.norm(shifted - shifted.mean(axis=0), axis=1)) < 0.5


def test_laplacian_policy_properties():
    env = ConsensusEnv(max_control=0.3)
    env.reset(seed=0)
    action = env.laplacian_policy(gain=2.0)
    assert action.dtype == np.float32 and action.shape == (8, 2)
    assert env.action_space.contains(action)
    # Symmetric graphs: the unclipped feedback sums to zero (centroid is invariant).
    env = ConsensusEnv(max_control=1e6)
    env.reset(seed=0)
    assert np.allclose(env.laplacian_policy().sum(axis=0), 0.0, atol=1e-3)
    with pytest.raises(ValueError):
        env.laplacian_policy(gain=0.0)


def test_laplacian_policy_warns_when_unstable():
    env = ConsensusEnv(topology="complete", n_agents=30, max_control=100.0)
    env.reset(seed=0)
    with pytest.warns(RuntimeWarning, match="unstable"):
        env.laplacian_policy(gain=1.0)


def test_double_integrator_wall_stops_agent():
    env = ConsensusEnv(dynamics="double", n_agents=2, arena_size=5.0, damping=0.0)
    env.reset(
        seed=0,
        options={"positions": [[4.99, 0.0], [0.0, 0.0]], "velocities": [[1.0, 0.5], [0.0, 0.0]]},
    )
    obs, *_ = env.step(np.zeros((2, 2)))
    assert env.positions[0, 0] == pytest.approx(5.0)
    assert env.velocities[0, 0] == 0.0 and env.velocities[0, 1] == pytest.approx(0.5)
    np.testing.assert_allclose(obs[0, 2:4], [0.0, 0.5])


def test_single_integrator_reports_realised_velocity():
    env = ConsensusEnv(n_agents=2, arena_size=5.0, dt=0.1)
    env.reset(seed=0, options={"positions": [[4.95, 0.0], [0.0, 0.0]]})
    env.step(np.array([[1.0, -1.0], [0.5, 0.5]]))
    np.testing.assert_allclose(env.velocities, [[0.5, -1.0], [0.5, 0.5]], atol=1e-12)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_agents": 1},
        {"task": "flocking"},
        {"topology": "grid"},
        {"dynamics": "triple"},
        {"dt": 0.0},
        {"max_steps": 0},
        {"arena_size": -1.0},
        {"max_control": 0.0},
        {"control_cost": -0.1},
        {"noise_std": -1.0},
        {"tolerance": 0.0},
        {"success_bonus": float("nan")},
        {"formation_shape": "hexagon"},
        {"formation_radius": 0.0},
        {"comm_radius": -2.0},
        {"edge_probability": 0.0},
        {"edge_probability": 1.5},
        {"graph_seed": -3},
        {"max_neighbors": 0},
        {"damping": -0.5},
        {"render_mode": "ansi"},
        {"task": "formation", "formation_radius": 12.0},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        ConsensusEnv(**kwargs)


@pytest.mark.parametrize("kwargs", [{"n_agents": 8.0}, {"max_steps": "10"}, {"dt": None}])
def test_wrong_argument_types_raise_type_error(kwargs):
    with pytest.raises(TypeError):
        ConsensusEnv(**kwargs)


def test_arguments_are_keyword_only():
    with pytest.raises(TypeError):
        ConsensusEnv(8)  # type: ignore[misc]


def test_action_and_reset_validation():
    env = ConsensusEnv()
    with pytest.raises(RuntimeError):
        env.step(np.zeros((8, 2)))
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros((8, 3)))
    with pytest.raises(ValueError):
        env.step(np.full((8, 2), np.nan))
    env.step(np.zeros(16))  # flat actions are accepted
    with pytest.raises(ValueError):
        env.reset(options={"positions": np.zeros((3, 2))})


def test_unknown_reset_options_warn_and_are_ignored():
    env = ConsensusEnv(n_agents=3)
    start = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.warns(UserWarning, match="ignoring unknown reset option") as record:
        obs, _ = env.reset(seed=0, options={"speed": 1, "positions": start})
    assert record[0].filename == __file__  # the warning points at the caller
    np.testing.assert_array_equal(env.positions, start)  # known keys still apply
    with pytest.warns(UserWarning, match="ignoring unknown reset option"):
        env.reset(seed=0, options={"options": 1})  # as passed by PettingZoo's API test
    with pytest.raises(ValueError), pytest.warns(UserWarning):  # known keys stay strict
        env.reset(options={"speed": 1, "positions": np.zeros((2, 2))})


def test_make_formation():
    for shape in FORMATION_SHAPES:
        offsets = make_formation(shape, 7, radius=2.0)
        assert offsets.shape == (7, 2)
        np.testing.assert_allclose(offsets.mean(axis=0), 0.0, atol=1e-12)
        distances = np.linalg.norm(offsets[:, None] - offsets[None], axis=-1)
        assert distances[np.triu_indices(7, 1)].min() > 0.1  # distinct slots
    circle = make_formation("circle", 6, radius=3.0)
    np.testing.assert_allclose(np.linalg.norm(circle, axis=1), 3.0)
    np.testing.assert_allclose(circle[0], [0.0, 3.0], atol=1e-12)
    with pytest.raises(ValueError):
        make_formation("hexagon", 4)


# ---------------------------------------------------------------------------
# Registry and Gymnasium API
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "env_id, task", [("Consensus-v0", "consensus"), ("Formation-v0", "formation")]
)
def test_make_registered(env_id, task):
    env = env_lib.make(env_id)
    assert isinstance(env.unwrapped, ConsensusEnv)
    assert env.unwrapped.task == task
    assert env_lib.ConsensusEnv is ConsensusEnv
    obs, _ = env.reset(seed=0)
    obs, reward, terminated, truncated, _ = env.step(env.unwrapped.laplacian_policy())
    assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"task": "formation", "topology": "proximity", "dynamics": "double", "noise_std": 0.1}],
)
def test_check_env(kwargs):
    env = ConsensusEnv(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The observation space is deliberately unbounded (relative positions/velocities).
        warnings.filterwarnings("ignore", message=r".*Box observation space m\w+ value is")
        check_env(env.unwrapped, skip_render_check=True)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_episode(env, steps=6):
    env.reset(seed=0)
    frames = [env.render()]
    for _ in range(steps):
        env.step(env.laplacian_policy())
        frames.append(env.render())
    return frames


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_rgb_array_frames(theme):
    previous = rendering.get_theme()
    rendering.set_theme(theme)
    try:
        env = ConsensusEnv(task="formation", render_mode="rgb_array")
        frames = _render_episode(env)
        env.close()
    finally:
        rendering.set_theme(previous)
    for frame in frames:
        assert frame.shape == (560, 1000, 3) and frame.dtype == np.uint8
        assert frame.std() > 5.0
    assert not np.array_equal(frames[0], frames[-1])
    background = frames[0][2, 2]
    expected = np.array(
        [int(rendering.get_theme(theme).background[k : k + 2], 16) for k in (1, 3, 5)]
    )
    np.testing.assert_allclose(background, expected, atol=2)


def test_blitted_frame_matches_full_redraw(light_theme):
    env = ConsensusEnv(topology="proximity", dynamics="double", render_mode="rgb_array")
    frames = _render_episode(env, steps=12)
    renderer = env._renderer
    for artist in renderer._dynamic_artists:
        artist.set_animated(False)
    full = figure_to_rgb(renderer.fig)
    diff = np.abs(frames[-1].astype(int) - full.astype(int))
    assert diff.mean() < 0.5
    env.close()


def test_render_after_reset_clears_history():
    env = ConsensusEnv(render_mode="rgb_array", max_steps=30)
    _render_episode(env, steps=10)
    env.reset(seed=1)
    env.render()
    renderer = env._renderer
    assert renderer._trail_count == 1
    assert np.count_nonzero(~np.isnan(renderer._error_hist)) == 1
    env.close()


def test_render_without_mode_warns_once():
    env = ConsensusEnv()
    env.reset(seed=0)
    with pytest.warns(UserWarning, match="render_mode"):
        assert env.render() is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert env.render() is None


def test_human_mode_runs_headless():
    env = ConsensusEnv(render_mode="human", max_steps=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # non-interactive Agg backend
        env.reset(seed=0)
        env.step(env.laplacian_policy())
        assert env.render() is None
    env.close()


def test_close_is_idempotent():
    env = ConsensusEnv(render_mode="rgb_array")
    env.close()
    env.reset(seed=0)
    env.render()
    env.close()
    env.close()
    frame = env.render()  # re-creates the figure
    assert frame.shape == (560, 1000, 3)
    env.close()
