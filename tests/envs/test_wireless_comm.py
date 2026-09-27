"""Tests for :class:`env_lib.wireless_comm_env.WirelessCommEnv`."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.utils import seeding
from gymnasium.utils.env_checker import check_env

import env_lib
from env_lib.utils import rendering
from env_lib.utils.rendering import figure_to_rgb
from env_lib.wireless_comm_env import WirelessCommEnv
from env_lib.wireless_comm_env.wireless_comm_env import (
    OUTCOME_COLLISION,
    OUTCOME_IDLE,
    OUTCOME_LOST,
    OUTCOME_SUCCESS,
)


# ---------------------------------------------------------------------------
# Straightforward loop reference of the original dynamics.
# ---------------------------------------------------------------------------
def reference_ap(i, j, action, gx, gy):
    if action == 0:
        return None
    di, dj = {1: (-1, -1), 2: (0, -1), 3: (-1, 0), 4: (0, 0)}[int(action)]
    x, y = i + di, j + dj
    if 0 <= x < gx - 1 and 0 <= y < gy - 1:
        return x, y
    return None


def reference_reset(rng, gx, gy, ddl, n):
    state = np.full((ddl, gx + 2 * n, gy + 2 * n), 2.0, dtype=np.float32)
    state[:, n : n + gx, n : n + gy] = rng.choice(2, size=(ddl, gx, gy))
    return state


def reference_step(state, action, rng, gx, gy, n, p, q):
    """One step of the original dynamics, agent by agent."""
    state = state.copy()
    profile = np.zeros((gx - 1, gy - 1), dtype=int)
    for k in range(gx * gy):
        ap = reference_ap(*divmod(k, gy), action[k], gx, gy)
        if ap is not None:
            profile[ap] += 1
    rewards = np.zeros(gx * gy)
    for k in range(gx * gy):
        i, j = divmod(k, gy)
        ap = reference_ap(i, j, action[k], gx, gy)
        queue = state[:, n + i, n + j]
        if action[k] != 0 and queue.max() == 1 and ap is not None and profile[ap] == 1:
            idx = 0
            while queue[idx] == 0:
                idx += 1
            if rng.random() <= q:
                state[idx, n + i, n + j] = 0
                rewards[k] = 1
    interior = state[:, n : n + gx, n : n + gy]
    interior[:-1] = interior[1:].copy()
    interior[-1] = rng.random((gx, gy)) <= p
    return state, rewards, profile


def reference_obs(state, gx, gy, n):
    w = 2 * n + 1
    return np.array(
        [
            state[:, i : i + w, j : j + w].reshape(-1)
            for i, j in (divmod(k, gy) for k in range(gx * gy))
        ],
        dtype=np.float32,
    )


CONFIGS = [
    dict(grid_x=6, grid_y=6),
    dict(grid_x=4, grid_y=3, ddl=3, n_obs_neighbors=0, packet_arrival_probability=0.5),
    dict(grid_x=2, grid_y=5, ddl=1, n_obs_neighbors=2, success_transmission_probability=1.0),
    dict(
        grid_x=5,
        grid_y=5,
        ddl=4,
        packet_arrival_probability=0.3,
        success_transmission_probability=0.2,
    ),
]


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
def test_default_spaces():
    env = WirelessCommEnv()
    assert env.n_agents == 36
    assert env.action_space == MultiDiscrete([5] * 36)
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (36, 2 * 9)
    assert env.observation_space.dtype == np.float32
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert set(info) >= {"agent_rewards", "outcomes", "ap_load"}


@pytest.mark.parametrize("config", CONFIGS)
def test_observation_layout_matches_reference(config):
    env = WirelessCommEnv(**config)
    gx, gy, n = env.grid_x, env.grid_y, env.n_obs_nghbr
    obs, _ = env.reset(seed=2)
    np.testing.assert_array_equal(obs, reference_obs(env.state, gx, gy, n))
    for _ in range(5):
        obs, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        np.testing.assert_array_equal(obs, reference_obs(env.state, gx, gy, n))
    # Observations are copies, not views of the internal state.
    obs[:] = -1
    assert env.state.min() >= 0


# ---------------------------------------------------------------------------
# Dynamics: equivalence with the loop reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config", CONFIGS)
def test_trajectory_matches_loop_reference(config):
    """The vectorised step draws the same random numbers in the same order."""
    env = WirelessCommEnv(max_iter=25, **config)
    gx, gy, n, ddl = env.grid_x, env.grid_y, env.n_obs_nghbr, env.ddl
    action_rng = np.random.default_rng(9)
    for seed in (0, 3):
        env.reset(seed=seed)
        ref_rng, _ = seeding.np_random(seed)
        state = reference_reset(ref_rng, gx, gy, ddl, n)
        np.testing.assert_array_equal(env.state, state)
        for t in range(25):
            action = action_rng.integers(5, size=gx * gy)
            obs, reward, terminated, truncated, info = env.step(action)
            state, rewards, profile = reference_step(
                state, action, ref_rng, gx, gy, n, env.p, env.q
            )
            np.testing.assert_array_equal(env.state, state)
            np.testing.assert_array_equal(obs, reference_obs(state, gx, gy, n))
            np.testing.assert_array_equal(info["agent_rewards"], rewards)
            np.testing.assert_array_equal(info["ap_load"], profile)
            assert reward == rewards.sum()
            assert terminated is False and truncated is (t == 24)


def test_deterministic_channel_per_step_semantics():
    """With q = 1 and p in {0, 1} every step is deterministic given the state."""
    for p in (0.0, 1.0):
        env = WirelessCommEnv(
            grid_x=5,
            grid_y=4,
            ddl=3,
            packet_arrival_probability=p,
            success_transmission_probability=1.0,
        )
        env.reset(seed=4)
        rng = np.random.default_rng(p == 1.0)
        for _ in range(20):
            action = rng.integers(5, size=env.n_agents) * (rng.random(env.n_agents) < 0.4)
            before = env.state.copy()
            _, reward, _, _, info = env.step(action)
            expected, rewards, _ = reference_step(
                before, action, np.random.default_rng(0), 5, 4, 1, p, 1.0
            )
            np.testing.assert_array_equal(env.state, expected)
            np.testing.assert_array_equal(info["agent_rewards"], rewards)


def test_collision_and_success_on_small_grid():
    env = WirelessCommEnv(
        grid_x=2,
        grid_y=2,
        ddl=2,
        packet_arrival_probability=1.0,
        success_transmission_probability=1.0,
    )
    env.reset(seed=0)
    env.state[:, 1:3, 1:3] = 1.0  # every agent holds two packets
    # All four agents target the single access point: 4-way collision.
    _, reward, _, _, info = env.step([4, 2, 3, 1])
    assert reward == 0.0
    np.testing.assert_array_equal(info["outcomes"], [OUTCOME_COLLISION] * 4)
    np.testing.assert_array_equal(info["ap_load"], [[4]])
    # One transmitter succeeds; out-of-range and idle agents do not interfere.
    _, reward, _, _, info = env.step([4, 0, 4, 0])
    assert reward == 1.0
    np.testing.assert_array_equal(
        info["outcomes"], [OUTCOME_SUCCESS, OUTCOME_IDLE, OUTCOME_LOST, OUTCOME_IDLE]
    )
    np.testing.assert_array_equal(info["agent_rewards"], [1.0, 0.0, 0.0, 0.0])


def test_earliest_packet_is_removed():
    env = WirelessCommEnv(
        grid_x=2,
        grid_y=2,
        ddl=3,
        packet_arrival_probability=0.0,
        success_transmission_probability=1.0,
    )
    env.reset(seed=0)
    env.state[:, 1, 1] = [0.0, 1.0, 1.0]
    env.step([4, 0, 0, 0])
    # Slot 1 was sent, slot 2 shifted into slot 1, no arrival.
    np.testing.assert_array_equal(env.state[:, 1, 1], [0.0, 1.0, 0.0])


def test_success_rate_matches_q():
    """Expected reward of a lone transmitter that always has a packet is q."""
    q = 0.35
    env = WirelessCommEnv(
        grid_x=2,
        grid_y=2,
        ddl=1,
        packet_arrival_probability=1.0,
        success_transmission_probability=q,
        max_iter=10**6,
    )
    env.reset(seed=123)
    env.state[:, 1:3, 1:3] = 1.0
    n = 4000
    total = sum(env.step([4, 0, 0, 0])[1] for _ in range(n))
    # 5 standard deviations of a binomial proportion.
    assert abs(total / n - q) < 5 * np.sqrt(q * (1 - q) / n)


def test_outcomes_are_consistent():
    env = WirelessCommEnv(grid_x=5, grid_y=5)
    env.reset(seed=1)
    for _ in range(30):
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        outcomes, rewards = info["outcomes"], info["agent_rewards"]
        assert outcomes.dtype == np.int8
        assert rewards.dtype == np.float64 and rewards.shape == (25,)
        assert set(np.unique(rewards)) <= {0.0, 1.0}
        np.testing.assert_array_equal(outcomes == OUTCOME_SUCCESS, rewards == 1.0)
        np.testing.assert_array_equal(outcomes == OUTCOME_IDLE, action == 0)
        assert reward == rewards.sum()
        assert info["ap_load"].sum() == np.sum(
            [reference_ap(*divmod(k, 5), a, 5, 5) is not None for k, a in enumerate(action)]
        )
        np.testing.assert_array_equal(env.last_action, action)


def test_actions_are_recorded():
    env = WirelessCommEnv(grid_x=3, grid_y=4, n_obs_neighbors=1)
    env.reset(seed=0)
    assert not env.actions.any()
    action = np.arange(12) % 5
    env.step(action)
    np.testing.assert_array_equal(env.actions[1:4, 1:5], action.reshape(3, 4))
    assert env.actions.sum() == action.sum()  # padding stays zero


def test_grid_shaped_actions_are_accepted():
    a, b = WirelessCommEnv(grid_x=3, grid_y=4), WirelessCommEnv(grid_x=3, grid_y=4)
    a.reset(seed=5)
    b.reset(seed=5)
    action = np.random.default_rng(0).integers(5, size=12)
    np.testing.assert_array_equal(a.step(action)[0], b.step(action.reshape(3, 4))[0])


def test_seeding_determinism():
    def rollout(seed):
        env = WirelessCommEnv(grid_x=4, grid_y=4)
        env.action_space.seed(seed)
        obs, _ = env.reset(seed=seed)
        trace = [obs]
        for _ in range(10):
            obs, reward, *_ = env.step(env.action_space.sample())
            trace.append(obs + reward)
        return np.stack(trace)

    np.testing.assert_array_equal(rollout(1), rollout(1))
    assert not np.array_equal(rollout(1), rollout(2))


def test_deprecated_seed_reseeds_generator():
    env = WirelessCommEnv(grid_x=3, grid_y=3)
    with pytest.warns(DeprecationWarning, match=r"reset\(seed=\.\.\.\)"):
        assert env.seed(8) == [8]
    expected, _ = seeding.np_random(8)
    assert env.np_random.random() == expected.random()


# ---------------------------------------------------------------------------
# Public helper methods
# ---------------------------------------------------------------------------
def test_access_point_mapping_matches_reference():
    env = WirelessCommEnv(grid_x=4, grid_y=5)
    for k in range(env.n_agents):
        i, j = divmod(k, 5)
        for action in range(5):
            expected = reference_ap(i, j, action, 4, 5)
            assert env.access_point_mapping(i, j, action) == (expected or (None, None))
    with pytest.raises(ValueError, match="not defined"):
        env.access_point_mapping(0, 0, 5)
    with pytest.raises(ValueError, match="outside"):
        env.access_point_mapping(4, 0, 1)


def test_check_transmission_fail():
    env = WirelessCommEnv(grid_x=3, grid_y=3)
    profile = np.array([[1, 2], [0, 1]])
    assert env.check_transmission_fail(None, None, profile) is True
    assert env.check_transmission_fail(0, 0, profile) is False
    assert env.check_transmission_fail(0, 1, profile) is True
    assert env.check_transmission_fail(1, 0, profile) is True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
def test_step_and_render_require_reset():
    env = WirelessCommEnv(render_mode="ansi")
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        env.step(np.zeros(36, dtype=int))
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        env.render()


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"grid_x": 1}, ValueError),
        ({"grid_y": 1}, ValueError),
        ({"grid_x": 2.5}, TypeError),
        ({"ddl": 0}, ValueError),
        ({"packet_arrival_probability": 1.5}, ValueError),
        ({"success_transmission_probability": -0.1}, ValueError),
        ({"success_transmission_probability": "high"}, TypeError),
        ({"n_obs_neighbors": -1}, ValueError),
        ({"max_iter": 0}, ValueError),
        ({"render_mode": "video"}, ValueError),
    ],
)
def test_invalid_arguments(kwargs, error):
    with pytest.raises(error):
        WirelessCommEnv(**kwargs)


@pytest.mark.parametrize(
    "action",
    [np.zeros(8, dtype=int), np.zeros(10, dtype=int), np.zeros((2, 4)), [5] * 9, [-1] + [0] * 8],
)
def test_invalid_actions(action):
    env = WirelessCommEnv(grid_x=3, grid_y=3)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_ansi_text():
    env = WirelessCommEnv(grid_x=2, grid_y=2, render_mode="ansi")
    env.reset(seed=0)
    env.step([1, 2, 3, 4])
    text = env.render()
    lines = text.splitlines()
    assert lines[0] == "Current state: "
    assert len(lines) == 5
    for k, action in enumerate([1, 2, 3, 4]):
        queue = env.state[:, 1 + k // 2, 1 + k % 2]
        assert lines[k + 1] == f"Agent {k}: action = {action}, state = {queue}"


def _check_frames(env, steps=4):
    env.reset(seed=0)
    frames = [env.render()]
    for _ in range(steps):
        env.step(env.action_space.sample())
        frames.append(env.render())
    for frame in frames:
        assert frame.shape == (560, 1000, 3) and frame.dtype == np.uint8
        assert frame.std() > 1.0
    assert not np.array_equal(frames[0], frames[-1])
    return frames


def test_rgb_array_dark():
    env = WirelessCommEnv(grid_x=4, grid_y=4, render_mode="rgb_array")
    frames = _check_frames(env)
    assert frames[0][2, 2].mean() < 60
    env.close()


def test_rgb_array_light(light_theme):
    env = WirelessCommEnv(grid_x=4, grid_y=4, ddl=3, render_mode="rgb_array")
    frames = _check_frames(env)
    assert frames[0][2, 2].mean() > 200
    env.close()


def test_blitted_frame_equals_full_redraw():
    env = WirelessCommEnv(grid_x=3, grid_y=4, max_iter=4, render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(9):  # past max_iter: the time axes scroll
        env.step(env.action_space.sample())
    frame = env.render()
    renderer = env._renderer
    for artist in renderer._dynamic_artists:
        artist.set_animated(False)
    np.testing.assert_array_equal(frame, figure_to_rgb(renderer.fig))
    env.close()


def test_human_mode_runs_headless():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # non-interactive backend
        env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="human")
        env.reset(seed=0)
        env.step(env.action_space.sample())
        assert env.render() is None
    env.close()


def test_close_is_idempotent():
    env = WirelessCommEnv(grid_x=3, grid_y=3, render_mode="rgb_array")
    env.reset(seed=0)
    env.render()
    env.close()
    env.close()
    assert env.render().shape == (560, 1000, 3)
    env.close()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_id, n_agents", [("WirelessComm-v0", 36), ("WirelessComm-v1", 16)])
def test_make_registered_ids(env_id, n_agents):
    with warnings.catch_warnings():
        # Gymnasium flags "-v0" as outdated because a "-v1" id (the 4x4 variant) exists.
        warnings.filterwarnings("ignore", message=".*is out of date.*")
        env = env_lib.make(env_id)
    assert isinstance(env.unwrapped, WirelessCommEnv)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == n_agents
    steps, truncated = 0, False
    while not truncated:
        *_, truncated, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == 50
    env.close()


def test_deprecated_register_shim():
    from env_lib import wireless_comm_env

    with pytest.warns(DeprecationWarning, match="register_envs"):
        wireless_comm_env.register()


def test_pickle_roundtrip():
    env = WirelessCommEnv(grid_x=3, grid_y=5, ddl=3, packet_arrival_probability=0.4)
    clone = pickle.loads(pickle.dumps(env))
    assert (clone.grid_x, clone.grid_y, clone.ddl, clone.p) == (3, 5, 3, 0.4)


def test_check_env():
    env = WirelessCommEnv(grid_x=3, grid_y=4)
    with warnings.catch_warnings():
        # The deprecated seed() method is kept for backwards compatibility.
        warnings.filterwarnings("ignore", message=".*`seed` function is dropped.*")
        check_env(env, skip_render_check=True)
