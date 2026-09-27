"""Tests for IPPO and MAPPO (marl_algorithms.algorithms.ppo)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import env_lib
from marl_algorithms import IPPO, MAPPO, PPOConfig, make_vector_env, train
from marl_algorithms.core.base import Algorithm
from marl_algorithms.core.buffers import RolloutBuffer
from marl_algorithms.presets.ppo import PRESETS
from marl_algorithms.registry import get_algorithm

ALGORITHMS = (IPPO, MAPPO)
GRID3 = {"n_buses": 3, "topology": "ring"}  # small power grid (3 agents, 10 features)

# Small instances of the environments (id, constructor arguments, n_agents, per-agent action shape).
CONTINUOUS = [
    ("PowerGrid-v0", {"n_buses": 4, "topology": "ring"}, 4, (1,)),
    ("Consensus-v0", {"n_agents": 3}, 3, (2,)),
]
DISCRETE = [
    ("LineMsg-v0", {"num_agents": 4}, 4, ()),
    ("WirelessComm-v1", {}, 16, ()),
]


def _envs(env_id: str, kwargs: dict, num_envs: int = 4):
    return make_vector_env(env_id, num_envs, **kwargs)


def _small(cls, envs, **overrides):
    """Algorithm with a tiny, fast configuration."""
    settings = dict(seed=0, hidden_sizes=(16,), rollout_length=8, n_epochs=2, num_minibatches=2)
    settings.update(overrides)
    return cls(envs, **settings)


def _parameters(algo: Algorithm) -> list[torch.Tensor]:
    return [p.detach().clone() for p in (*algo.actor.parameters(), *algo.critic.parameters())]


# ---------------------------------------------------------------------------
# Construction, acting and shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", ALGORITHMS)
@pytest.mark.parametrize(("env_id", "kwargs", "n", "action_shape"), CONTINUOUS + DISCRETE)
def test_act_shapes_and_bounds(cls, env_id, kwargs, n, action_shape):
    envs = _envs(env_id, kwargs)
    algo = _small(cls, envs)
    obs, _ = envs.reset(seed=0)
    agent_obs = algo.spec.agent_obs(obs)
    for deterministic in (False, True):
        actions = algo.act(agent_obs, deterministic=deterministic)
        assert actions.shape == (4, n, *action_shape)
        if algo.spec.continuous:
            assert actions.dtype == np.float32
            assert np.all(actions >= algo.spec.action_low) and np.all(
                actions <= algo.spec.action_high
            )
        else:
            assert actions.dtype == np.int64
            assert actions.min() >= 0 and actions.max() < algo.spec.n_actions
        assert envs.action_space.contains(algo.spec.env_action(actions))
    # Extra leading batch dimensions are allowed.
    assert algo.act(agent_obs[None], deterministic=True).shape == (1, 4, n, *action_shape)
    envs.close()


@pytest.mark.parametrize("cls", ALGORITHMS)
@pytest.mark.parametrize(("env_id", "kwargs"), [(e, k) for e, k, _, _ in CONTINUOUS + DISCRETE])
def test_policy_on_single_env_with_evaluate(cls, env_id, kwargs):
    if env_id.startswith("LineMsg"):
        kwargs = {**kwargs, "action_space_type": "multibinary"}
    env = env_lib.make(env_id, **kwargs)
    algo = _small(cls, env)
    result = env_lib.evaluate(env, algo.policy(), n_episodes=2, seed=0, max_steps=20)
    assert result.n_episodes == 2 and np.isfinite(result.mean_return)
    env.close()


def test_default_config_and_reward_source():
    envs = _envs("PowerGrid-v0", GRID3)
    assert IPPO(envs).reward_source == "agent"
    assert MAPPO(envs).reward_source == "team"
    assert MAPPO(envs, reward_source="agent").reward_source == "agent"
    assert IPPO(envs, reward_source="team").reward_source == "team"
    assert isinstance(MAPPO(envs).config, PPOConfig)
    assert get_algorithm("ippo") is IPPO and get_algorithm("mappo") is MAPPO


def test_critic_inputs_centralised_vs_decentralised():
    """MAPPO values depend on every agent's observation, IPPO values only on the own one."""
    envs = _envs("PowerGrid-v0", GRID3)
    obs = torch.randn(5, 3, 10)
    changed = obs.clone()
    changed[:, 2] += 1.0  # perturb agent 2 only
    for cls, affects_others in ((IPPO, False), (MAPPO, True)):
        for shared in (True, False):
            algo = cls(envs, seed=0, share_parameters=shared)
            with torch.no_grad():
                delta = (algo.values(changed) - algo.values(obs)).abs()
            assert delta.shape == (5, 3)
            assert bool((delta[:, :2] > 0).any()) is affects_others
            assert bool((delta[:, 2] > 0).all())
    mappo = MAPPO(envs)
    assert mappo.critic_input_dim == 3 * 10 + 3  # global state and agent id
    assert MAPPO(envs, share_parameters=False).critic_input_dim == 3 * 10
    assert IPPO(envs).critic_input_dim == 10 + 3


def test_heterogeneous_action_bounds():
    u_max = np.array([0.05, 0.1, 0.2], dtype=np.float32)
    envs = _envs("PowerGrid-v0", {**GRID3, "u_max": u_max})
    obs = np.random.default_rng(0).normal(size=(64, 3, 10)).astype(np.float32) * 50.0
    for shared in (True, False):
        algo = MAPPO(envs, seed=0, share_parameters=shared, log_std_init=2.0)
        actions = algo.act(obs)[..., 0]
        assert np.all(np.abs(actions) <= u_max + 1e-7)
        assert np.allclose(np.abs(actions).max(axis=0), u_max)  # saturates at each bound


def test_rollout_log_prob_is_that_of_the_stored_sample():
    """Continuous samples are stored unclipped with their exact log-probability."""
    envs = _envs("PowerGrid-v0", GRID3)
    algo = IPPO(envs, seed=0, log_std_init=1.0)
    obs = algo.spec.agent_obs(envs.reset(seed=0)[0])
    actions, extras = algo._rollout_step(obs)
    assert np.any(np.abs(actions) > algo.spec.action_high[..., 0].max())  # unclipped
    inputs = algo.agent_inputs(algo.tensor(extras["norm_obs"]))
    with torch.no_grad():
        log_prob, _ = algo.actor.call("log_prob", inputs, algo.tensor(actions))
    np.testing.assert_allclose(log_prob.numpy(), extras["log_prob"], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Wrong usage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", ALGORITHMS)
def test_wrong_usage_raises(cls):
    envs = _envs("PowerGrid-v0", GRID3)
    # A joint Discrete(2 ** n) action cannot be split into per-agent actions.
    with pytest.raises(TypeError):
        cls(env_lib.make("LineMsg-v0", num_agents=4))
    with pytest.raises(TypeError, match="unknown PPOConfig fields"):
        cls(envs, learning_rate=1e-3)
    with pytest.raises(TypeError, match="config must be"):
        cls(envs, config=object())
    algo = cls(envs)
    with pytest.raises(ValueError, match="expected observations"):
        algo.act(np.zeros((2, 4, 10), dtype=np.float32))
    next_step = env_lib.make_vec("PowerGrid-v0", 2, **GRID3)  # "next_step" autoreset
    with pytest.raises(ValueError, match="same_step"):
        algo.learn(next_step, 10)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_epochs": 0},
        {"num_minibatches": 0},
        {"clip_range": 0.0},
        {"value_clip_range": -0.1},
        {"huber_delta": 0.0},
        {"ent_coef": -0.01},
        {"critic_lr": 0.0},
        {"target_kl": 0.0},
        {"critic_hidden_sizes": ()},
        {"reward_source": "global"},
        {"lr": 0.0},
        {"gae_lambda": 1.5},
    ],
)
def test_config_validation(overrides):
    with pytest.raises(ValueError):
        PPOConfig(**overrides)


# ---------------------------------------------------------------------------
# Learning loop
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", ALGORITHMS)
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize(("env_id", "kwargs"), [("Consensus-v0", {"n_agents": 3}), DISCRETE[0][:2]])
def test_short_learn(cls, shared, env_id, kwargs):
    envs = _envs(env_id, kwargs)
    algo = _small(cls, envs, share_parameters=shared, reward_source="agent" if shared else "team")
    before = _parameters(algo)
    log = algo.learn(envs, 64, seed=0)
    assert algo.env_steps == 64 and algo.num_updates == 2 and len(log.updates) == 2
    assert any(not torch.equal(a, b) for a, b in zip(before, _parameters(algo)))
    stats = log.updates[-1][1]
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        assert np.isfinite(stats[key]), key
    assert np.isfinite(stats["explained_variance"])
    assert stats["approx_kl"] >= 0.0 and 0.0 <= stats["clip_fraction"] <= 1.0
    envs.close()


def test_update_statistics_and_options():
    envs = _envs("PowerGrid-v0", GRID3)
    algo = _small(
        MAPPO, envs, anneal_lr=True, huber_delta=1.0, value_clip_range=None, target_kl=1e-9
    )
    log = algo.learn(envs, 96, seed=0)
    rates = [stats["learning_rate"] for _, stats in log.updates]
    assert rates[0] == pytest.approx(algo.config.lr) and rates[1] < rates[0] and rates[2] < rates[1]
    assert algo.actor_optimizer.param_groups[0]["lr"] == pytest.approx(rates[-1])
    # The tiny KL target stops every update after its first epoch.
    assert all(stats["epochs"] == 1.0 for _, stats in log.updates)
    assert "action_std" in log.updates[-1][1] and "reward_std" in log.updates[-1][1]


def test_value_targets_are_learned():
    """With a fixed policy the critic fits the returns (explained variance grows)."""
    envs = _envs("LineMsg-v0", {"num_agents": 4}, num_envs=8)
    algo = IPPO(
        envs,
        seed=0,
        rollout_length=25,
        lr=1e-9,  # (practically) frozen actor
        critic_lr=3e-3,
        gamma=0.8,
        hidden_sizes=(32, 32),
        normalize_rewards=False,
    )
    log = algo.learn(envs, 8 * 25 * 8, seed=0)
    variance = [stats["explained_variance"] for _, stats in log.updates]
    assert np.all(np.isfinite(variance))
    assert variance[0] < 0.5 < min(variance[-3:])


@pytest.mark.parametrize("cls", ALGORITHMS)
@pytest.mark.parametrize(("env_id", "kwargs"), [CONTINUOUS[0][:2], DISCRETE[0][:2]])
def test_seed_determinism(cls, env_id, kwargs):
    results = []
    for _ in range(2):
        envs = _envs(env_id, kwargs)
        algo = _small(cls, envs)
        log = algo.learn(envs, 64, seed=3)
        results.append((_parameters(algo), log.updates[-1][1]))
        envs.close()
    (first, stats_a), (second, stats_b) = results
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert stats_a == stats_b
    envs = _envs(env_id, kwargs)
    other = _small(cls, envs, seed=1)
    other.learn(envs, 64, seed=3)
    assert any(not torch.equal(a, b) for a, b in zip(first, _parameters(other)))


@pytest.mark.parametrize("cls", ALGORITHMS)
@pytest.mark.parametrize("shared", [True, False])
def test_save_load_round_trip(cls, shared, tmp_path):
    envs = _envs("PowerGrid-v0", GRID3)
    algo = _small(cls, envs, share_parameters=shared, reward_source="agent")
    algo.learn(envs, 64, seed=0)
    path = algo.save(tmp_path / "ppo.pt")
    loaded = Algorithm.load(path)
    assert type(loaded) is cls and loaded.config == algo.config
    assert loaded.env_steps == algo.env_steps and loaded.num_updates == algo.num_updates
    obs = np.random.default_rng(0).normal(size=(6, 3, 10)).astype(np.float32)
    np.testing.assert_array_equal(algo.act(obs, True), loaded.act(obs, True))
    np.testing.assert_array_equal(algo.obs_normalizer.rms.mean, loaded.obs_normalizer.rms.mean)
    assert loaded.reward_scaler is not None
    assert float(loaded.reward_scaler.rms.var) == float(algo.reward_scaler.rms.var)
    with torch.no_grad():
        x = torch.randn(6, 3, 10)
        torch.testing.assert_close(algo.values(x), loaded.values(x), rtol=0, atol=0)
    # Training can continue after loading.
    loaded.learn(envs, 96, seed=1)
    assert loaded.env_steps == 96


def test_train_entry_point_and_presets():
    algo, log = train(
        "mappo",
        "LineMsg-v0",
        32,
        num_envs=2,
        env_kwargs={"num_agents": 3},
        hidden_sizes=(8,),
        rollout_length=16,
    )
    assert isinstance(algo, MAPPO) and log.env_id == "LineMsg-v0" and algo.env_steps == 32
    assert set(PRESETS) == {"ippo", "mappo"}
    assert {"PowerGrid-v0", "Platoon-v0", "Consensus-v0", "LineMsg-v0"} <= set(PRESETS["mappo"])
    assert "PowerGrid-v0" in PRESETS["ippo"]
    registered = set(env_lib.list_envs())
    for presets in PRESETS.values():
        for env_id, preset in presets.items():
            assert env_id in registered
            assert set(preset) == {"num_envs", "total_steps", "config", "env_kwargs"}
            config = dataclasses.replace(PPOConfig(), **preset["config"])
            assert isinstance(config, PPOConfig)
            assert preset["num_envs"] >= 1 and preset["total_steps"] >= preset["num_envs"]


# ---------------------------------------------------------------------------
# Learning sanity
# ---------------------------------------------------------------------------
def test_mappo_learns_to_relay_messages():
    """MAPPO learns the LineMsg relay (all agents keep their link) within a few thousand steps."""
    envs = _envs("LineMsg-v0", {"num_agents": 4}, num_envs=16)
    algo = MAPPO(envs, seed=0, rollout_length=16, lr=3e-3, ent_coef=0.0, hidden_sizes=(32,))
    random_return = env_lib.evaluate(envs, None, n_episodes=16, seed=1).mean_return
    algo.learn(envs, 6_000, seed=0)
    trained = algo.evaluate(envs, n_episodes=16, seed=1).mean_return
    baseline = env_lib.evaluate(envs, env_lib.baseline_policy(envs), n_episodes=16, seed=1)
    assert trained > random_return + 0.8 * (baseline.mean_return - random_return)


@pytest.mark.parametrize("cls", ALGORITHMS)
@pytest.mark.parametrize("source", ["team", "agent"])
def test_training_rewards_follow_reward_source(cls, source):
    """The critics learn the team reward (broadcast to all agents) or each agent's own reward."""
    envs = _envs("PowerGrid-v0", GRID3)
    rng = np.random.default_rng(0)
    fields = {
        "reward": ((), np.float32),
        "agent_rewards": ((3,), np.float32),
        "terminated": ((), bool),
        "truncated": ((), bool),
    }
    rollout = RolloutBuffer(5, 4, fields)
    for _ in range(5):
        agent_rewards = rng.normal(size=(4, 3))
        rollout.add(
            reward=agent_rewards.sum(axis=1),
            agent_rewards=agent_rewards,
            terminated=np.zeros(4, bool),
            truncated=rng.random(4) < 0.3,
        )
    expected = rollout["reward"][..., None] if source == "team" else rollout["agent_rewards"]
    raw = cls(envs, reward_source=source, normalize_rewards=False)._training_rewards(rollout)
    assert raw.shape == (5, 4, 3)
    np.testing.assert_allclose(raw, np.broadcast_to(expected, (5, 4, 3)), rtol=1e-6)
    scaled = cls(envs, reward_source=source)._training_rewards(rollout)
    factor = scaled / raw  # one positive scale per step, shared by copies and agents
    assert np.all(factor > 0)
    np.testing.assert_allclose(factor, factor[:, :1, :1] * np.ones_like(factor), rtol=1e-6)
