"""Tests for the Pistonball environment (env_lib.pistonball_env)."""

from __future__ import annotations

import pickle
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

pytest.importorskip("pymunk")

import env_lib
from env_lib.pistonball_env import ManualPolicy, PistonballEnv
from env_lib.utils import rendering as theming

WALL, RADIUS, WIDTH, BALL_R = 80, 5, 40, 40


def reference_index(x, n):
    """Piston under ``x`` as written in the original implementation."""
    return max(0, min(int((x - WALL - RADIUS) / WIDTH), n - 1))


def reference_local_rewards(env, prev_left, curr_left, ball_x, ball_vx, prev_y, new_y):
    """Straightforward per-piston loop of the Pistonball reward rule."""
    n = env.n_pistons
    hit = (ball_x - BALL_R + ball_vx * env.dt) <= WALL + 0.5
    rewards = np.zeros(n)
    for i in range(n):
        sees_prev = abs(i - reference_index(prev_left, n)) <= env.kappa
        sees_curr = abs(i - reference_index(curr_left, n)) <= env.kappa
        if sees_prev or sees_curr:
            if prev_left > curr_left:
                ball = 0.5 * (prev_left - curr_left)
            else:
                ball = prev_left - curr_left
            r = ball + env.time_penalty
        else:
            r = env.time_penalty
        if env.terminated_condition and hit:
            if abs(i - reference_index(ball_x, n)) <= env.kappa:
                r += env.termination_reward
            if i == 0:
                r += env.leftmost_piston_reward
        if env.movement_penalty != 0.0:
            moved = abs(new_y[i] - prev_y[i])
            if moved > env.movement_penalty_threshold:
                r += env.movement_penalty * (moved / env.pixels_per_position)
        rewards[i] = r
    return rewards


def heuristic_action(env, rng=None, explore=0.0):
    """Raise the pistons right of the ball and lower the others (continuous)."""
    action = np.where(env._piston_x > env.ball.position[0] - 20, 1.0, -1.0).astype(np.float32)
    if rng is not None and explore > 0:
        noisy = rng.random(env.n_pistons) < explore
        action[noisy] = rng.uniform(-1, 1, noisy.sum())
    return action


def run_until_goal(env, seed, other=None, attempts=10):
    """Drive the ball left with the heuristic until an episode terminates.

    The heuristic occasionally traps the ball, so a few seeds are tried; ``other``
    (an environment with identical physics) receives the same actions.
    """
    for attempt in range(attempts):
        env.reset(seed=seed + attempt)
        if other is not None:
            other.reset(seed=seed + attempt)
        for _ in range(env.max_cycles):
            action = heuristic_action(env)
            out = env.step(action)
            out_other = other.step(action) if other is not None else None
            if out[2]:
                return out, out_other
            if out[3]:
                break
    raise AssertionError("the heuristic policy never reached the goal")


# ---------------------------------------------------------------------------
# Spaces and observations
# ---------------------------------------------------------------------------
def test_continuous_spaces():
    env = PistonballEnv(n_pistons=6)
    assert env.action_space.shape == (6,)
    assert env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    assert env.observation_space.shape == (6, 7)
    assert env.observation_space.dtype == np.float32
    assert env.screen_width == 160 + 40 * 6


def test_discrete_spaces_are_multidiscrete():
    from gymnasium.spaces import MultiDiscrete

    env = PistonballEnv(n_pistons=5, continuous=False)
    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [3] * 5
    env.reset(seed=0)
    for action in ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [0, 1, 2, 1, 0]):
        obs, *_ = env.step(np.array(action))
        assert env.observation_space.contains(obs)


@pytest.mark.parametrize("continuous", [True, False])
def test_observations_float32_and_contained(continuous):
    env = PistonballEnv(n_pistons=8, continuous=continuous, kappa=2)
    obs, info = env.reset(seed=1)
    assert info == {}
    env.action_space.seed(1)
    for _ in range(60):
        assert obs.dtype == np.float32 and obs.shape == (8, 7)
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        if terminated or truncated:
            obs, _ = env.reset()


def test_observation_structure():
    n = 10
    env = PistonballEnv(n_pistons=n, kappa=1)
    obs, _ = env.reset(seed=4)
    expected_x = (5 + 40 * np.arange(n)) / (40 * n)
    np.testing.assert_allclose(obs[:, 1], expected_x.astype(np.float32))
    assert np.all(np.diff(obs[:, 1]) > 0)
    assert np.all((obs[:, 0] >= -1) & (obs[:, 0] <= 1))
    np.testing.assert_allclose(
        obs[:, 0], ((env.piston_pos_y - env.mid_piston_y) / 32).astype(np.float32)
    )
    x, y = env.ball.position
    vx, vy = env.ball.velocity
    expected_ball = np.array(
        [
            (x - 80) / (env.screen_width - 160),
            (y - 80) / (env.screen_height - 160),
            vx / 15,
            vy / 8,
            env.ball.angular_velocity / 8,
        ],
        dtype=np.float32,
    )
    mask = env.observable_mask()
    np.testing.assert_array_equal(obs[mask, 2:], np.tile(expected_ball, (mask.sum(), 1)))
    assert np.all(obs[~mask, 2:] == 0)
    obs2, *_ = env.step(np.zeros(n))
    np.testing.assert_array_equal(obs2[:, 1], obs[:, 1])


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def rollout(env, seed, steps=40):
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    trace = [obs]
    rewards = []
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        trace.append(obs)
        rewards.append(info["local_rewards"])
        if terminated or truncated:
            break
    return np.stack(trace), np.stack(rewards)


def test_seeding_determinism():
    a, b = PistonballEnv(n_pistons=6), PistonballEnv(n_pistons=6)
    obs_a, rew_a = rollout(a, seed=7)
    obs_b, rew_b = rollout(b, seed=7)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(rew_a, rew_b)
    obs_c, _ = rollout(a, seed=8)
    assert not np.array_equal(obs_a[0], obs_c[0])
    # Re-seeding the same instance reproduces the episode.
    obs_d, _ = rollout(a, seed=7)
    np.testing.assert_array_equal(obs_a, obs_d)


def test_deprecated_seed_method():
    env = PistonballEnv(n_pistons=4)
    with pytest.warns(DeprecationWarning, match="reset\\(seed"):
        assert env.seed(3) == [3]
    obs_a, _ = env.reset()
    obs_b, _ = PistonballEnv(n_pistons=4).reset(seed=3)
    np.testing.assert_array_equal(obs_a, obs_b)


# ---------------------------------------------------------------------------
# Observability (kappa-hop neighbourhood)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [0, 1, 2, 3, 25])
def test_kappa_mask_matches_reference(kappa):
    n = 9
    env = PistonballEnv(n_pistons=n, kappa=kappa)
    env.reset(seed=0)
    for x in np.linspace(-50, env.screen_width + 50, 157):
        idx = reference_index(x, n)
        expected = np.array([abs(i - idx) <= kappa for i in range(n)])
        np.testing.assert_array_equal(env.observable_mask(x), expected)
        assert [env._can_observe_ball(i, x) for i in range(n)] == expected.tolist()

    for x in (130.0, 200.0, 333.3, 420.0, 470.0):
        env.ball.position = (x, env.ball.position[1])
        obs = env._get_obs()
        expected = env.observable_mask(x)
        idx = reference_index(x, n)
        np.testing.assert_array_equal(np.any(obs[:, 2:] != 0, axis=1), expected)
        assert expected.sum() == min(idx + kappa, n - 1) - max(idx - kappa, 0) + 1


# ---------------------------------------------------------------------------
# Rewards and dynamics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "config",
    [
        dict(n_pistons=10, kappa=1),
        dict(n_pistons=8, kappa=2, movement_penalty=-0.05, movement_penalty_threshold=1.0),
        dict(n_pistons=7, kappa=0, leftmost_piston_reward=1.0, termination_reward=2.0),
        dict(n_pistons=9, continuous=False, movement_penalty=-0.1, time_penalty=-0.3),
        dict(n_pistons=6, terminated_condition=False, kappa=1),
    ],
)
def test_local_rewards_match_reference_rollout(config):
    env = PistonballEnv(**config)
    rng = np.random.default_rng(0)
    terminal_steps = 0
    for seed in range(3):
        env.reset(seed=seed)
        prev_left = int(env.ball.position[0] - BALL_R)
        for _ in range(env.max_cycles):
            action = heuristic_action(env, rng, explore=0.3)
            if not env.continuous:
                action = np.rint(action).astype(np.int64) + 1
                expected_move = action - 1.0
            else:
                expected_move = np.clip(action, -1, 1).astype(np.float64)
            prev_y = env.piston_pos_y.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            new_y = env.piston_pos_y.copy()
            np.testing.assert_array_equal(
                new_y,
                np.clip(prev_y - 4 * expected_move, env.minimum_piston_y, env.maximum_piston_y),
            )
            x = env.ball.position[0]
            curr_left = int(x - BALL_R)
            expected = reference_local_rewards(
                env, prev_left, curr_left, x, env.ball.velocity[0], prev_y, new_y
            )
            np.testing.assert_allclose(info["local_rewards"], expected, rtol=0, atol=1e-12)
            assert info["agent_rewards"].dtype == np.float64
            assert info["agent_rewards"].shape == (env.n_pistons,)
            np.testing.assert_array_equal(info["agent_rewards"], info["local_rewards"])
            assert reward == pytest.approx(expected.sum(), abs=1e-9)
            assert info["total_reward"] == reward
            assert env.last_ball_x == curr_left
            np.testing.assert_array_equal(
                env.last_ball_positions, np.full(env.n_pistons, curr_left)
            )
            prev_left = curr_left
            terminal_steps += terminated
            if terminated or truncated:
                break
    if env.terminated_condition:
        assert terminal_steps > 0, "rollouts never reached the goal; reward branch untested"


def test_get_local_reward_for_piston_matches_vectorised():
    env = PistonballEnv(n_pistons=12, kappa=2)
    env.reset(seed=0)
    for prev_x, curr_x in [(700, 690), (690, 700), (300, 300), (95, 81), (1000, 60)]:
        scalar = [env.get_local_reward_for_piston(i, prev_x, curr_x) for i in range(12)]
        np.testing.assert_array_equal(env._local_rewards(prev_x, curr_x), scalar)


def test_movement_penalty_threshold():
    kwargs = dict(n_pistons=6, kappa=0)
    plain = PistonballEnv(**kwargs)
    penalised = PistonballEnv(**kwargs, movement_penalty=-0.1, movement_penalty_threshold=3.0)
    plain.reset(seed=2)
    penalised.reset(seed=2)
    for value, penalty in [(1.0, -0.1), (0.5, 0.0), (-0.25, 0.0), (0.0, 0.0)]:
        action = np.zeros(6, dtype=np.float32)
        action[1] = value  # moves piston 1 by 4 * |value| pixels
        _, _, _, _, info_plain = plain.step(action)
        _, _, _, _, info_pen = penalised.step(action)
        diff = info_pen["local_rewards"] - info_plain["local_rewards"]
        expected = np.zeros(6)
        expected[1] = penalty
        np.testing.assert_allclose(diff, expected, atol=1e-12)


def test_actions_are_clipped():
    a, b = PistonballEnv(n_pistons=5), PistonballEnv(n_pistons=5)
    a.reset(seed=0)
    b.reset(seed=0)
    obs_a, *_ = a.step(np.array([2.0, -2.0, 1.5, -1.5, 0.5]))
    obs_b, *_ = b.step(np.array([1.0, -1.0, 1.0, -1.0, 0.5]))
    np.testing.assert_array_equal(obs_a, obs_b)


def test_discrete_matches_continuous_extremes():
    d = PistonballEnv(n_pistons=4, continuous=False)
    c = PistonballEnv(n_pistons=4)
    d.reset(seed=5)
    c.reset(seed=5)
    for step in range(20):
        discrete = np.array([0, 1, 2, (step % 3)])
        out_d = d.step(discrete)
        out_c = c.step((discrete - 1).astype(np.float32))
        np.testing.assert_array_equal(out_d[0], out_c[0])
        assert out_d[1] == out_c[1]


# ---------------------------------------------------------------------------
# Termination and truncation
# ---------------------------------------------------------------------------
def test_termination_rewards_at_left_wall():
    kwargs = dict(
        n_pistons=6, kappa=1, termination_reward=1.0, leftmost_piston_reward=0.5, max_cycles=300
    )
    env = PistonballEnv(**kwargs)
    free = PistonballEnv(**kwargs, terminated_condition=False)  # identical physics
    (_, reward, terminated, truncated, info), other = run_until_goal(env, seed=0, other=free)
    _, reward_free, terminated_free, _, info_free = other
    assert terminated and not truncated and not terminated_free
    x = env.ball.position[0]
    assert x - BALL_R + env.ball.velocity[0] * env.dt <= WALL + 0.5
    observers = env.observable_mask()
    assert observers[0] and not observers[-1]
    expected = 1.0 * observers
    expected[0] += 0.5
    bonus = info["local_rewards"] - info_free["local_rewards"]
    np.testing.assert_allclose(bonus, expected, atol=1e-12)
    assert reward == pytest.approx(reward_free + expected.sum())
    # The flag is sticky until the next reset.
    _, _, terminated, _, _ = env.step(np.zeros(6))
    assert terminated
    env.reset(seed=1)
    assert env.terminate is False


@pytest.mark.parametrize("n_pistons", [5, 10])
def test_heuristic_policy_reaches_goal(n_pistons):
    env = PistonballEnv(n_pistons=n_pistons, kappa=2, max_cycles=300)
    (_, _, terminated, truncated, _), _ = run_until_goal(env, seed=0)
    assert terminated and not truncated
    assert env.frames < env.max_cycles


def test_truncation_at_max_cycles():
    env = PistonballEnv(n_pistons=5, max_cycles=5, terminated_condition=False)
    env.reset(seed=0)
    for step in range(1, 6):
        _, _, terminated, truncated, _ = env.step(np.zeros(5))
        assert not terminated
        assert truncated == (step == 5)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("continuous", [True, False])
@pytest.mark.parametrize("shape", [(3,), (6,), (1, 5), ()])
def test_invalid_action_shape_raises(continuous, shape):
    env = PistonballEnv(n_pistons=5, continuous=continuous)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="Action shape"):
        env.step(np.ones(shape))


def test_invalid_action_values_raise():
    env = PistonballEnv(n_pistons=4, continuous=False)
    env.reset(seed=0)
    for bad in ([0, 1, 2, 3], [0, 1, 2, -1]):
        with pytest.raises(ValueError, match="0, 1, 2"):
            env.step(np.array(bad))
    cont = PistonballEnv(n_pistons=4)
    cont.reset(seed=0)
    with pytest.raises(ValueError, match="finite"):
        cont.step(np.array([0.0, np.nan, 0.0, 0.0]))


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError, match="reset"):
        PistonballEnv(n_pistons=3).step(np.zeros(3))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(n_pistons=0), ValueError),
        (dict(n_pistons=2.5), TypeError),
        (dict(kappa=-1), ValueError),
        (dict(kappa=1.5), TypeError),
        (dict(max_cycles=0), ValueError),
        (dict(ball_mass=0.0), ValueError),
        (dict(ball_friction=-1.0), ValueError),
        (dict(render_mode="window"), ValueError),
        (dict(time_penalty="a"), TypeError),
    ],
)
def test_constructor_validation(kwargs, error):
    with pytest.raises(error):
        PistonballEnv(**kwargs)


def test_default_arguments_are_preserved():
    env = PistonballEnv()
    assert env.n_pistons == 20 and env.time_penalty == -0.1 and env.continuous
    assert env.random_drop and env.random_rotate
    assert (env.ball_mass, env.ball_friction, env.ball_elasticity) == (0.75, 0.3, 1.5)
    assert env.max_cycles == 125 and env.render_mode is None
    assert env.movement_penalty == 0.0 and env.movement_penalty_threshold == 0.01
    assert env.kappa == 1 and env.terminated_condition
    assert env.leftmost_piston_reward == 0.0 and env.termination_reward == 0.5
    assert env.agents[0] == "piston_0" and env.agent_name_mapping["piston_19"] == 19


# ---------------------------------------------------------------------------
# Gymnasium integration
# ---------------------------------------------------------------------------
def test_make_registered_id():
    env = env_lib.make("Pistonball-v0", n_pistons=6)
    assert isinstance(env.unwrapped, PistonballEnv)
    assert env.unwrapped.n_pistons == 6
    obs, _ = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (6, 7) and "agent_rewards" in info
    env.close()


@pytest.mark.parametrize("continuous", [True, False])
def test_check_env(continuous):
    from gymnasium.utils.env_checker import check_env

    env = PistonballEnv(n_pistons=5, continuous=continuous)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the checker must not find anything else to warn about
        # The observation bounds are (-inf, inf) by design (and unchanged from the original
        # API): the ball velocities are not bounded, so finite bounds could be violated.
        warnings.filterwarnings("ignore", message=".*Box observation space m(in|ax)imum value")
        # seed() is kept (deprecated) for backwards compatibility.
        warnings.filterwarnings("ignore", message=".*support for the `seed` function is dropped")
        check_env(env.unwrapped, skip_render_check=True)


def test_pickle_roundtrip():
    env = PistonballEnv(n_pistons=5, kappa=2, movement_penalty=-0.2, continuous=False)
    env.reset(seed=0)
    clone = pickle.loads(pickle.dumps(env))
    assert (clone.n_pistons, clone.kappa, clone.movement_penalty) == (5, 2, -0.2)
    assert not clone.continuous
    np.testing.assert_array_equal(env.reset(seed=9)[0], clone.reset(seed=9)[0])


def test_deprecated_drawing_methods_warn():
    env = PistonballEnv(n_pistons=3)
    for name in ("draw", "draw_background", "draw_pistons", "enable_render"):
        with pytest.warns(DeprecationWarning, match="render"):
            getattr(env, name)()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_no_rendering_work_without_render_call():
    env = PistonballEnv(n_pistons=5, render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    assert env.renderer is None
    headless = PistonballEnv(n_pistons=5)
    headless.reset(seed=0)
    headless.step(np.zeros(5))
    with pytest.warns(UserWarning, match="render_mode"):
        assert headless.render() is None
    assert headless.renderer is None


def test_headless_training_never_imports_pygame():
    code = textwrap.dedent(
        """
        import sys
        import numpy as np
        import env_lib
        env = env_lib.make("Pistonball-v0", n_pistons=6)
        env.reset(seed=0)
        for _ in range(20):
            env.step(env.action_space.sample())
        env.close()
        assert "pygame" not in sys.modules, "pygame was imported"
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


@pytest.fixture
def restore_theme():
    previous = theming.get_theme()
    yield
    theming.set_theme(previous)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_rgb_array_frames(theme, restore_theme):
    pytest.importorskip("pygame")
    theming.set_theme(theme)
    env = PistonballEnv(n_pistons=7, render_mode="rgb_array", kappa=1)
    env.reset(seed=0)
    first = env.render()
    assert first.shape == (560, 160 + 40 * 7, 3)
    assert first.dtype == np.uint8
    assert first.std() > 10
    for _ in range(5):
        env.step(heuristic_action(env))
    second = env.render()
    assert second.shape == first.shape
    assert not np.array_equal(first, second)
    corner = first[:5, :5].mean()
    assert corner < 80 if theme == "dark" else corner > 180
    env.close()


def test_render_after_termination_and_reset():
    pytest.importorskip("pygame")
    env = PistonballEnv(n_pistons=5, render_mode="rgb_array", max_cycles=300)
    (_, _, terminated, _, _), _ = run_until_goal(env, seed=0)
    assert terminated
    assert env.render().shape == (560, 360, 3)
    env.reset(seed=1)
    assert len(env.renderer._trail) == 0
    assert env.render().shape == (560, 360, 3)


def test_close_is_idempotent():
    pytest.importorskip("pygame")
    env = PistonballEnv(n_pistons=4, render_mode="rgb_array")
    env.close()
    env.reset(seed=0)
    env.render()
    assert env.renderer is not None
    env.close()
    env.close()
    assert env.renderer is None
    assert env.render().shape == (560, 320, 3)  # a new renderer is created on demand
    env.close()


def test_human_mode_with_dummy_driver():
    pygame = pytest.importorskip("pygame")
    env = PistonballEnv(n_pistons=4, render_mode="human")
    env.reset(seed=0)  # renders automatically
    assert env.renderer is not None and env.renderer.is_open
    assert env.render() is None
    env.step(np.zeros(4))
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    env.step(np.zeros(4))  # the renderer handles QUIT by closing its window
    assert not env.renderer.is_open
    assert env.render() is None
    env.close()
    env.close()


def test_manual_policy_neutral_actions():
    pytest.importorskip("pygame")
    env = PistonballEnv(n_pistons=4)
    policy = ManualPolicy(env, agent_id=2)
    action = policy(np.zeros((4, 7)))
    assert action.shape == (4,) and np.all(action == 0.0)
    assert policy.available_agents == env.agents and policy.running
    discrete = ManualPolicy(PistonballEnv(n_pistons=4, continuous=False))
    assert discrete(None).tolist() == [1, 1, 1, 1]
    with pytest.raises(ValueError):
        ManualPolicy(env, agent_id=4)
