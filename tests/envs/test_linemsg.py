"""Tests for :class:`env_lib.linemsg_env.LineMsgEnv`."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete, MultiBinary
from gymnasium.utils import seeding
from gymnasium.utils.env_checker import check_env

import env_lib
from env_lib.linemsg_env import LineMsgEnv
from env_lib.utils import rendering
from env_lib.utils.rendering import figure_to_rgb


# ---------------------------------------------------------------------------
# Straightforward loop reference of the original dynamics.
# ---------------------------------------------------------------------------
def reference_decode(action: int, num_agents: int) -> np.ndarray:
    return np.array([(action // (2**i)) % 2 for i in range(num_agents)])


def reference_step(states: np.ndarray, bits: np.ndarray, rng: np.random.Generator):
    """One step of the original LineMsg dynamics written agent by agent."""
    num_agents = len(states)
    u = rng.random(num_agents - 2)
    new = np.zeros(num_agents, dtype=int)
    new[0] = states[1]
    for i in range(1, num_agents - 1):
        if bits[i] == 1 and states[i + 1] == 1:
            new[i] = 1
        elif bits[i] == 1 and states[i + 1] == 0 and u[i - 1] < 0.8:
            new[i] = 1
    new[-1] = bits[-1]
    rewards = np.array([0.1 * (s == 1) for s in new])
    rewards[0] *= 10
    return new, rewards


def reference_obs(states: np.ndarray, n_nb: int) -> np.ndarray:
    padded = np.concatenate([[2] * n_nb, states, [2] * n_nb])
    return np.array([padded[i : i + 2 * n_nb + 1] for i in range(len(states))], dtype=np.float32)


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
    env = LineMsgEnv()
    assert env.action_space == Discrete(2**10)
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (10, 3)
    assert env.observation_space.dtype == np.float32
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == (10, 3)
    assert env.observation_space.contains(obs)
    assert info["agent_rewards"].shape == (10,)


@pytest.mark.parametrize("num_agents, n_nb", [(3, 1), (7, 2), (12, 3)])
def test_observation_window_matches_reference(num_agents, n_nb):
    env = LineMsgEnv(num_agents=num_agents, n_obs_neighbors=n_nb)
    assert env.observation_space.shape == (num_agents, 2 * n_nb + 1)
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs, *_ = env.step(rng.integers(2, size=num_agents))
        assert env.observation_space.contains(obs)
        agent_states = env.state[n_nb : n_nb + num_agents]
        np.testing.assert_array_equal(obs, reference_obs(agent_states, n_nb))


def test_zero_neighbours_is_treated_as_one():
    with pytest.warns(UserWarning, match="n_obs_neighbors=0"):
        env = LineMsgEnv(num_agents=5, n_obs_neighbors=0)
    assert env.n_obs_nghbr == 1
    assert env.observation_space.shape == (5, 3)


def test_state_and_action_layout():
    env = LineMsgEnv(num_agents=4, n_obs_neighbors=2)
    env.reset(seed=0)
    np.testing.assert_array_equal(env.state, [2, 2, 1, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(env.actions, [2] * 8)
    env.step([1, 0, 1, 1])
    np.testing.assert_array_equal(env.actions, [2, 2, 1, 0, 1, 1, 2, 2])


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_agents, n_nb", [(3, 1), (10, 1), (9, 3)])
def test_trajectory_matches_loop_reference(num_agents, n_nb):
    env = LineMsgEnv(num_agents=num_agents, n_obs_neighbors=n_nb, max_iter=40)
    action_rng = np.random.default_rng(42)
    for seed in (0, 7):
        env.reset(seed=seed)
        ref_rng, _ = seeding.np_random(seed)
        states = np.ones(num_agents, dtype=int)
        for t in range(40):
            action = int(action_rng.integers(2**num_agents))
            obs, reward, terminated, truncated, info = env.step(action)
            states, rewards = reference_step(states, reference_decode(action, num_agents), ref_rng)
            np.testing.assert_array_equal(env.state[n_nb : n_nb + num_agents], states)
            np.testing.assert_array_equal(obs, reference_obs(states, n_nb))
            np.testing.assert_allclose(info["agent_rewards"], rewards, rtol=0, atol=1e-12)
            assert reward == pytest.approx(rewards.sum(), abs=1e-12)
            assert terminated is False
            assert truncated is (t == 39)


def test_integer_and_array_actions_are_equivalent():
    num_agents = 8
    a = LineMsgEnv(num_agents=num_agents)
    b = LineMsgEnv(num_agents=num_agents)
    a.reset(seed=3)
    b.reset(seed=3)
    rng = np.random.default_rng(1)
    for _ in range(30):
        action = int(rng.integers(2**num_agents))
        bits = reference_decode(action, num_agents)
        out_a = a.step(np.int64(action))
        out_b = b.step(bits.tolist())
        np.testing.assert_array_equal(out_a[0], out_b[0])
        assert out_a[1] == out_b[1]
    # Boolean arrays and 0-d integer arrays are accepted as well.
    a.step(np.ones(num_agents, dtype=bool))
    a.step(np.array(5))


def test_rewards_and_info():
    env = LineMsgEnv(num_agents=6)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step([1] * 6)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    agent_rewards = info["agent_rewards"]
    assert agent_rewards.dtype == np.float64 and agent_rewards.shape == (6,)
    assert agent_rewards.sum() == pytest.approx(reward)
    # Everybody active and holding the message: sink 1.0, others 0.1 each.
    assert reward == pytest.approx(1.0 + 0.1 * 5)
    obs, reward, *_ = env.step([0] * 6)
    # Only the sink (copies agent 1, which held the message) keeps it.
    np.testing.assert_array_equal(env.state[1:7], [1, 0, 0, 0, 0, 0])
    assert reward == pytest.approx(1.0)


def test_truncation_after_max_iter():
    env = LineMsgEnv(num_agents=4, max_iter=5)
    env.reset(seed=0)
    flags = [env.step(env.action_space.sample())[3] for _ in range(5)]
    assert flags == [False] * 4 + [True]


def test_seeding_determinism():
    def rollout(seed):
        env = LineMsgEnv(num_agents=6)
        env.action_space.seed(seed)
        obs, _ = env.reset(seed=seed)
        trace = [obs]
        for _ in range(15):
            obs, reward, *_ = env.step(env.action_space.sample())
            trace.append(obs * reward)
        return np.stack(trace)

    np.testing.assert_array_equal(rollout(5), rollout(5))
    assert not np.array_equal(rollout(5), rollout(6))


def test_deprecated_seed_reseeds_generator():
    env = LineMsgEnv(num_agents=5)
    with pytest.warns(DeprecationWarning, match=r"reset\(seed=\.\.\.\)"):
        assert env.seed(11) == [11]
    expected, _ = seeding.np_random(11)
    assert env.np_random.random() == expected.random()


# ---------------------------------------------------------------------------
# Action spaces
# ---------------------------------------------------------------------------
def test_multibinary_mode():
    env = LineMsgEnv(num_agents=7, action_space_type="multibinary")
    assert env.action_space == MultiBinary(7)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.step(3)  # integer actions are still accepted
    np.testing.assert_array_equal(env.actions[1:8], [1, 1, 0, 0, 0, 0, 0])


def test_large_lines_need_multibinary():
    with pytest.raises(ValueError, match="multibinary"):
        LineMsgEnv(num_agents=63)
    env = LineMsgEnv(num_agents=100, action_space_type="multibinary")
    obs, _ = env.reset(seed=0)
    obs, *_ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    with pytest.raises(ValueError, match="num_agents <= 62"):
        env.step(1)


def test_discrete_limit_is_62_agents():
    env = LineMsgEnv(num_agents=62)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.step(2**62 - 1)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
def test_step_and_render_require_reset():
    env = LineMsgEnv(render_mode="ansi")
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        env.step(0)
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        env.render()


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"num_agents": 2}, ValueError),
        ({"num_agents": 4.0}, TypeError),
        ({"n_obs_neighbors": -1}, ValueError),
        ({"max_iter": 0}, ValueError),
        ({"action_space_type": "binary"}, ValueError),
        ({"render_mode": "video"}, ValueError),
    ],
)
def test_invalid_arguments(kwargs, error):
    with pytest.raises(error):
        LineMsgEnv(**kwargs)


@pytest.mark.parametrize(
    "action",
    [-1, 2**5, [0, 1, 1], [0, 1, 2, 0, 1], [0.5, 0, 0, 0, 0], np.zeros((5, 1)), 1.0],
)
def test_invalid_actions(action):
    env = LineMsgEnv(num_agents=5)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_ansi_text():
    env = LineMsgEnv(num_agents=3, render_mode="ansi")
    env.reset(seed=0)
    assert env.render() == (
        "Current state: \n"
        "Agent 0: action = 2, state = 1\n"
        "Agent 1: action = 2, state = 1\n"
        "Agent 2: action = 2, state = 1\n"
    )
    env.step([0, 1, 0])
    text = env.render()
    assert "Agent 1: action = 1, state = 1" in text
    assert "Agent 2: action = 0, state = 0" in text


def test_render_without_mode_returns_none():
    env = LineMsgEnv()
    env.reset(seed=0)
    with pytest.warns(UserWarning, match="render_mode"):
        assert env.render() is None


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
    env = LineMsgEnv(num_agents=8, render_mode="rgb_array")
    _check_frames(env)
    env.close()


def test_rgb_array_light(light_theme):
    env = LineMsgEnv(num_agents=8, render_mode="rgb_array")
    frames = _check_frames(env)
    # Light theme: the page background (top-left pixel) is bright.
    assert frames[0][2, 2].mean() > 200
    env.close()


def test_blitted_frame_equals_full_redraw():
    env = LineMsgEnv(num_agents=6, max_iter=4, render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(9):  # past max_iter: the raster window scrolls
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
        env = LineMsgEnv(num_agents=5, render_mode="human")
        env.reset(seed=0)
        env.step(env.action_space.sample())
        assert env.render() is None
    env.close()


def test_close_is_idempotent():
    env = LineMsgEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.render()
    env.close()
    env.close()
    # Rendering after close re-creates the figure.
    assert env.render().shape == (560, 1000, 3)
    env.close()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
def test_make_registered_id():
    env = env_lib.make("LineMsg-v0")
    assert isinstance(env.unwrapped, LineMsgEnv)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (10, 3)
    steps = 0
    truncated = False
    while not truncated:
        *_, truncated, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == 50
    env.close()


def test_pickle_roundtrip():
    env = LineMsgEnv(num_agents=5, n_obs_neighbors=2, max_iter=7, action_space_type="multibinary")
    clone = pickle.loads(pickle.dumps(env))
    assert (clone.num_agents, clone.n_obs_nghbr, clone.max_iter) == (5, 2, 7)
    assert clone.action_space == MultiBinary(5)


@pytest.mark.parametrize("action_space_type", ["discrete", "multibinary"])
def test_check_env(action_space_type):
    env = LineMsgEnv(num_agents=6, action_space_type=action_space_type)
    with warnings.catch_warnings():
        # The deprecated seed() method is kept for backwards compatibility.
        warnings.filterwarnings("ignore", message=".*`seed` function is dropped.*")
        check_env(env, skip_render_check=True)
