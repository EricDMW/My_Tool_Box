"""Reward modes of the Kuramoto and AJLATT environments."""

from __future__ import annotations

import numpy as np
import pytest

import env_lib
from env_lib import AJLATTEnv
from env_lib.kos_env import KuramotoOscillatorEnv, KuramotoOscillatorVectorEnv
from env_lib.kos_env._common import REWARD_MODES

N = 5
BONUS = 3.0


def synced_state(n=N):
    return {"phases": np.full(n, 0.3), "natural_frequencies": np.full(n, 1.0)}


def spread_state(n=N):
    return {
        "phases": np.linspace(-np.pi, np.pi, n, endpoint=False),
        "natural_frequencies": np.linspace(0.5, 2.0, n),
    }


def kuramoto(**kwargs):
    kwargs = {"n_oscillators": N, "coupling_mode": "constant", "sync_bonus": BONUS, **kwargs}
    return KuramotoOscillatorEnv(**kwargs)


def zeros():
    return np.zeros(N, dtype=np.float32)


# ---------------------------------------------------------------------------
# Kuramoto
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", REWARD_MODES)
def test_kuramoto_mode_on_a_synchronising_step(mode):
    env = kuramoto(reward_mode=mode)
    _, info = env.reset(seed=0, options=synced_state())
    r0 = info["order_parameter"]
    _, reward, terminated, truncated, info = env.step(zeros())
    r = info["order_parameter"]
    assert terminated and not truncated and info["synchronized"]
    expected = {
        "dense": r + BONUS,
        "penalty": r - 1.0 + BONUS,
        "progress": r - r0 + BONUS,
        "terminal": r + BONUS,  # the step ends the episode
        "sparse": BONUS,
    }[mode]
    assert reward == pytest.approx(expected)


@pytest.mark.parametrize("mode", REWARD_MODES)
def test_kuramoto_mode_on_an_ordinary_step(mode):
    env = kuramoto(reward_mode=mode, coupling_strength=0.0)
    _, info = env.reset(seed=0, options=spread_state())
    r0 = info["order_parameter"]
    _, reward, terminated, truncated, info = env.step(zeros())
    r = info["order_parameter"]
    assert not (terminated or truncated or info["synchronized"])
    expected = {"dense": r, "penalty": r - 1.0, "progress": r - r0, "terminal": 0.0, "sparse": 0.0}
    assert reward == pytest.approx(expected[mode])
    if mode == "penalty":
        assert reward <= 0


def test_kuramoto_terminal_reward_is_paid_on_the_last_step_only():
    env = kuramoto(reward_mode="terminal", coupling_strength=0.0, max_steps=4)
    env.reset(seed=0, options=spread_state())
    steps = [env.step(zeros()) for _ in range(4)]
    rewards = [step[1] for step in steps]
    assert rewards[:3] == [0.0, 0.0, 0.0]
    assert steps[-1][3] and rewards[-1] == pytest.approx(steps[-1][4]["order_parameter"])


def test_kuramoto_progress_return_telescopes():
    env = kuramoto(reward_mode="progress", coupling_strength=2.0, max_steps=60)
    _, info = env.reset(seed=0, options=spread_state())
    start, total, bonuses = info["order_parameter"], 0.0, 0
    rng = np.random.default_rng(0)
    while True:
        _, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, N).astype(np.float32))
        total += reward
        bonuses += info["synchronized"]
        if terminated or truncated:
            break
    assert total == pytest.approx(info["order_parameter"] - start + BONUS * bonuses)


def test_kuramoto_progress_with_frequency_reward_starts_from_the_initial_signal():
    env = kuramoto(reward_mode="progress", reward_type="frequency_synchronization")
    env.reset(seed=0, options=spread_state())
    initial = -np.mean(np.abs(spread_state()["natural_frequencies"] - 1.0))  # zero control
    _, reward, _, _, info = env.step(zeros())
    signal = -np.mean(np.abs(info["dphases_dt"] - 1.0))
    assert reward == pytest.approx(signal - initial)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_kuramoto_without_sync_termination_pays_every_synchronised_step(mode):
    env = kuramoto(reward_mode=mode, terminate_on_sync=False, max_steps=5)
    env.reset(seed=0, options=synced_state())
    for _ in range(5):
        _, reward, terminated, truncated, info = env.step(zeros())
        assert not terminated and info["synchronized"]
        expected = info["order_parameter"] + BONUS if mode == "dense" else BONUS
        assert reward == pytest.approx(expected)
    assert truncated


def test_kuramoto_terminal_without_sync_termination_waits_for_the_limit():
    env = kuramoto(reward_mode="terminal", terminate_on_sync=False, max_steps=3)
    env.reset(seed=0, options=synced_state())
    rewards = [env.step(zeros())[1] for _ in range(3)]
    assert rewards[:2] == [0.0, 0.0] and rewards[2] > BONUS


def test_kuramoto_control_cost():
    action = np.full(N, 0.5, dtype=np.float32)
    rewards = []
    for cost in (0.0, 2.0):
        env = kuramoto(control_cost=cost, coupling_strength=0.0)
        env.reset(seed=0, options=spread_state())
        rewards.append(env.step(action)[1])
    assert rewards[1] == pytest.approx(rewards[0] - 2.0 * 0.25)


def test_kuramoto_default_is_the_original_reward():
    env = kuramoto()
    assert (env.reward_mode, env.terminate_on_sync, env.control_cost) == ("dense", True, 0.0)
    env.reset(seed=0, options=spread_state())
    _, reward, _, _, info = env.step(zeros())
    assert reward == info["order_parameter"]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"reward_mode": "shaped"}, ValueError),
        ({"reward_mode": "sparse", "reward_type": "frequency_synchronization"}, ValueError),
        ({"control_cost": -1.0}, ValueError),
        ({"terminate_on_sync": "yes"}, TypeError),
    ],
)
def test_kuramoto_invalid_reward_settings(kwargs, error):
    with pytest.raises(error):
        kuramoto(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward_mode": "penalty", "control_cost": 0.1},
        {"reward_mode": "progress", "coupling_mode": "dynamic"},
        {"reward_mode": "progress", "reward_type": "frequency_synchronization"},
        {"reward_mode": "terminal", "sync_threshold": 0.7},
        {"reward_mode": "sparse", "terminate_on_sync": False, "sync_threshold": 0.7},
    ],
)
def test_kuramoto_vector_copies_match_single_env(kwargs):
    kwargs = {"n_oscillators": 6, "max_steps": 30, **kwargs}
    envs = KuramotoOscillatorVectorEnv(3, autoreset_mode="same_step", **kwargs)
    rng = np.random.default_rng(4)
    state = {
        "phases": rng.uniform(-np.pi, np.pi, (3, 6)),
        "natural_frequencies": rng.uniform(0.5, 2.0, (3, 6)),
    }
    if envs.coupling_mode == "dynamic":  # the initial gains enter the initial signal
        state["coupling_strengths"] = rng.uniform(0.0, 5.0, (3, envs.n_couplings))
    envs.reset(seed=0, options=state)
    singles = [KuramotoOscillatorEnv(**kwargs) for _ in range(3)]
    for b, single in enumerate(singles):
        single.reset(options={key: value[b] for key, value in state.items()})
    done = np.zeros(3, dtype=bool)
    for _ in range(30):
        actions = rng.uniform(envs.action_space.low, envs.action_space.high).astype(np.float32)
        _, rewards, terminated, truncated, infos = envs.step(actions)
        for b, single in enumerate(singles):
            if done[b]:
                continue
            _, reward, term, trunc, info = single.step(actions[b])
            assert rewards[b] == reward
            assert (terminated[b], truncated[b]) == (term, trunc)
            assert infos["synchronized"][b] == info["synchronized"]
            done[b] = term or trunc
        if done.all():
            break


@pytest.mark.parametrize("mode", REWARD_MODES)
def test_kuramoto_torch_backend_matches_numpy(mode):
    pytest.importorskip("torch")
    from env_lib.kos_env import KuramotoOscillatorEnvTorch

    kwargs = {
        "n_oscillators": N,
        "coupling_mode": "constant",
        "coupling_strength": 1.5,
        "reward_mode": mode,
        "sync_threshold": 0.9,
        "control_cost": 0.05,
        "max_steps": 25,
    }
    if mode == "sparse":
        kwargs["control_cost"] = 0.0
    numpy_env = KuramotoOscillatorEnv(**kwargs)
    torch_env = KuramotoOscillatorEnvTorch(n_agents=2, **kwargs)
    numpy_env.reset(seed=0, options=spread_state())
    torch_env.reset(seed=0, options=spread_state())
    rng = np.random.default_rng(1)
    for _ in range(25):
        action = rng.uniform(-1, 1, N).astype(np.float32)
        _, reward, term, trunc, info = numpy_env.step(action)
        _, t_reward, t_term, t_trunc, t_info = torch_env.step(action)
        assert t_reward == pytest.approx(reward, abs=1e-4)
        np.testing.assert_allclose(t_info["agent_rewards"], reward, atol=1e-4)
        assert (t_term, t_trunc) == (term, trunc)
        assert bool(t_info["synchronized"][0]) == info["synchronized"]
        if term or trunc:
            break


def test_kuramoto_modes_through_make_and_make_vec():
    env = env_lib.make("KuramotoOscillator-v1", reward_mode="penalty", terminate_on_sync=False)
    assert env.unwrapped.reward_mode == "penalty" and not env.unwrapped.terminate_on_sync
    envs = env_lib.make_vec("KuramotoOscillator-v1", 2, reward_mode="terminal")
    assert envs.reward_mode == "terminal"
    envs.reset(seed=0)
    _, rewards, *_ = envs.step(envs.action_space.sample())
    assert rewards.shape == (2,)
    envs.close()


# ---------------------------------------------------------------------------
# AJLATT
# ---------------------------------------------------------------------------
def hold_still():
    return np.zeros((4, 2))


def wall_position(env):
    grid = env.MAP
    cy, cx = np.argwhere(grid.map[20:-20, 20:-20] == 1)[0] + 20
    return grid.cell_to_se2([cx, cy])


def collide_robot_0(env):
    wall = wall_position(env)
    env.robot_est[0].state = np.array([wall[0] - 0.15, wall[1], 0.0])


def test_ajlatt_cost_mode_is_the_original_reward():
    env = AJLATTEnv(sigma_vR=0.0, sigma_wR=0.0)
    assert env.config.reward_mode == "cost"
    env.reset(seed=0)
    _, reward, _, _, info = env.step(hold_still())
    assert not info["collisions"].any()
    np.testing.assert_allclose(reward, -info["tracking_cost"], rtol=1e-12)


def test_ajlatt_bounded_mode():
    env = AJLATTEnv(reward_mode="bounded", cost_scale=10.0, sigma_vR=0.0, sigma_wR=0.0)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(hold_still())
    np.testing.assert_allclose(reward, np.exp(-info["tracking_cost"] / 10.0))
    assert np.all((reward > 0) & (reward <= 1))

    collide_robot_0(env)
    _, reward, terminated, _, info = env.step(hold_still())
    assert info["collisions"][0] and terminated[0]
    expected = np.exp(-info["tracking_cost"][0] / 10.0) - env.config.obstacle_penalty / 10.0
    assert reward[0] == pytest.approx(expected)


@pytest.mark.parametrize("terminate", [True, False])
def test_ajlatt_collision_termination_penalty(terminate):
    rewards = []
    for extra in (0.0, 50.0):
        env = AJLATTEnv(
            terminate_on_collision=terminate,
            collision_termination_penalty=extra,
            sigma_vR=0.0,
            sigma_wR=0.0,
        )
        env.reset(seed=0)
        collide_robot_0(env)
        _, reward, terminated, _, info = env.step(hold_still())
        assert info["collisions"][0] and terminated[0] == terminate
        rewards.append(reward)
    charged = 50.0 if terminate else 0.0
    assert rewards[1][0] == pytest.approx(rewards[0][0] - charged)
    np.testing.assert_array_equal(rewards[1][1:], rewards[0][1:])


def test_ajlatt_team_reward_weight():
    rewards = {}
    for weight in (0.0, 0.5, 1.0):
        env = AJLATTEnv(team_reward_weight=weight, sigma_vR=0.0, sigma_wR=0.0)
        env.reset(seed=0)
        collide_robot_0(env)  # make the robots' rewards differ
        rewards[weight] = env.step(hold_still())[1]
    individual = rewards[0.0]
    np.testing.assert_allclose(rewards[1.0], np.full(4, individual.mean()))
    np.testing.assert_allclose(rewards[0.5], 0.5 * individual + 0.5 * individual.mean())
    for reward in rewards.values():
        assert reward.sum() == pytest.approx(individual.sum())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward_mode": "cost_to_go"},
        {"cost_scale": 0.0},
        {"collision_termination_penalty": -1.0},
        {"team_reward_weight": 1.5},
    ],
)
def test_ajlatt_invalid_reward_settings(kwargs):
    with pytest.raises(ValueError):
        AJLATTEnv(**kwargs)


def test_ajlatt_bounded_mode_through_make_vec():
    envs = env_lib.make_vec("AJLATT-v0", 2, reward_mode="bounded", max_episode_steps=5)
    envs.reset(seed=0)
    _, rewards, *_ = envs.step(np.zeros(envs.action_space.shape, dtype=np.float32))
    assert rewards.shape == (2,) and np.all(rewards <= 4.0)  # team sum of four robots
    envs.close()
