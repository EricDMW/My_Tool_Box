"""Tests of the value-based algorithms IQL, VDN and QMIX."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import env_lib
from marl_algorithms import (
    IQL,
    QMIX,
    VDN,
    Algorithm,
    QLearningConfig,
    make_vector_env,
)
from marl_algorithms.algorithms.value_decomposition import PRESETS, VDNMixer
from marl_algorithms.core.networks import QMixer
from marl_algorithms.registry import get_algorithm, train

ALGORITHMS = [IQL, VDN, QMIX]
#: Small, fast settings for the unit tests.
FAST = {"hidden_sizes": (32, 32), "batch_size": 32, "warmup_steps": 64, "epsilon_decay_steps": 400}


def linemsg(num_envs: int = 4, num_agents: int = 4):
    return make_vector_env("LineMsg-v0", num_envs, num_agents=num_agents)


def wireless(num_envs: int = 4):
    return make_vector_env("WirelessComm-v1", num_envs)


def hand_batch(algo, size: int = 6, seed: int = 0) -> dict[str, np.ndarray]:
    """A replay-like batch with random contents and some terminal transitions."""
    rng = np.random.default_rng(seed)
    n, d, n_actions = algo.spec.n_agents, algo.spec.obs_dim, algo.spec.n_actions
    return {
        "obs": rng.random((size, n, d)).astype(np.float32),
        "actions": rng.integers(0, n_actions, size=(size, n)),
        "reward": rng.normal(size=size).astype(np.float32),
        "agent_rewards": rng.normal(size=(size, n)).astype(np.float32),
        "next_obs": rng.random((size, n, d)).astype(np.float32),
        "terminated": (np.arange(size) % 3 == 0).astype(np.float32),
    }


def as_tensors(algo, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        name: algo.tensor(value, torch.long if name == "actions" else torch.float32)
        for name, value in batch.items()
    }


def perturb(module: torch.nn.Module, seed: int = 1, scale: float = 0.5) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.add_(scale * torch.randn(param.shape, generator=generator))


# ----------------------------------------------------------------------
# Construction and configuration
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", ALGORITHMS)
@pytest.mark.parametrize(
    "make_env, n_agents, n_actions",
    [(linemsg, 4, 2), (wireless, 16, 5)],
    ids=["linemsg", "wireless"],
)
def test_construction_on_discrete_envs(cls, make_env, n_agents, n_actions):
    envs = make_env()
    try:
        algo = cls(envs, seed=0, **FAST)
    finally:
        envs.close()
    assert algo.spec.n_agents == n_agents and algo.spec.n_actions == n_actions
    assert len(algo.q_net.nets) == 1  # shared parameters by default
    assert algo.agent_input_dim == algo.spec.obs_dim + n_agents  # one-hot agent ids
    obs = torch.zeros(3, n_agents, algo.spec.obs_dim)
    assert algo.q_values(obs).shape == (3, n_agents, n_actions)
    if cls is IQL:
        assert algo.mixer is None and "mixer" not in algo._modules()
    else:
        assert isinstance(algo.mixer, VDNMixer if cls is VDN else QMixer)
        assert not any(p.requires_grad for p in algo.target_mixer.parameters())
    assert not any(p.requires_grad for p in algo.target_q_net.parameters())
    expected = {"q_net", "target_q_net", "optimizer"}
    if cls is not IQL:
        expected |= {"mixer", "target_mixer"}
    assert set(algo._modules()) == expected


@pytest.mark.parametrize("cls", ALGORITHMS)
def test_continuous_env_raises_type_error(cls):
    env = env_lib.make("Consensus-v0")
    try:
        with pytest.raises(TypeError, match="discrete"):
            cls(env)
    finally:
        env.close()


def test_registry_and_config():
    assert get_algorithm("iql") is IQL
    assert get_algorithm("VDN") is VDN
    assert get_algorithm("qmix") is QMIX
    assert all(cls.config_class is QLearningConfig for cls in ALGORITHMS)
    for bad in (
        {"loss": "l1"},
        {"epsilon_start": 0.1, "epsilon_end": 0.5},
        {"epsilon_end": -0.1},
        {"epsilon_decay_steps": -1},
        {"target_update_interval": -1},
        {"lr": 0.0},
        {"mixer_embed_dim": 0},
    ):
        with pytest.raises(ValueError):
            QLearningConfig(**bad)
    envs = linemsg()
    with pytest.raises(TypeError, match="unknown"):
        VDN(envs, not_a_field=1)
    envs.close()


# ----------------------------------------------------------------------
# Acting
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", ALGORITHMS)
def test_action_shapes_and_policy(cls):
    envs = wireless()
    algo = cls(envs, seed=0, **FAST)
    obs = algo.spec.agent_obs(envs.reset(seed=0)[0])
    envs.close()
    for deterministic in (True, False):
        actions = algo.act(obs, deterministic=deterministic)
        assert actions.shape == (4, 16) and actions.dtype == np.int64
        assert actions.min() >= 0 and actions.max() < 5
    # Arbitrary leading batch dimensions.
    assert algo.act(np.zeros((2, 3, 16, 18), np.float32), deterministic=True).shape == (2, 3, 16)

    env = env_lib.make("WirelessComm-v1")
    result = env_lib.evaluate(env, algo.policy(), n_episodes=2, seed=0)
    env.close()
    assert result.returns.shape == (2,) and np.all(np.isfinite(result.returns))


def test_policy_on_single_linemsg_env():
    envs = linemsg()
    algo = QMIX(envs, seed=0, **FAST)
    envs.close()
    env = env_lib.make("LineMsg-v0", num_agents=4, action_space_type="multibinary", max_iter=10)
    obs, _ = env.reset(seed=0)
    action = algo.policy()(obs)
    assert action.shape == (4,) and env.action_space.contains(action)
    result = env_lib.evaluate(env, algo.policy(), n_episodes=2, seed=0)
    env.close()
    assert np.all(result.lengths == 10)


def test_epsilon_schedule():
    envs = linemsg()
    algo = VDN(envs, seed=0, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=1000)
    envs.close()
    assert algo.epsilon_at(0) == pytest.approx(1.0)
    assert algo.epsilon_at(500) == pytest.approx(0.55)
    assert algo.epsilon_at(1000) == pytest.approx(0.1)
    assert algo.epsilon_at(10**6) == pytest.approx(0.1)
    algo.env_steps = 250
    assert algo.epsilon == pytest.approx(0.775)
    algo.config = dataclasses.replace(algo.config, epsilon_decay_steps=0)
    assert algo.epsilon_at(0) == pytest.approx(0.1)

    # Exploratory actions: greedy with epsilon 0, uniform over the choices with epsilon 1.
    obs = np.random.default_rng(0).random((2000, 4, 3)).astype(np.float32)
    greedy = algo.act(obs, deterministic=True)
    algo.config = dataclasses.replace(algo.config, epsilon_start=0.0, epsilon_end=0.0)
    np.testing.assert_array_equal(algo.act(obs), greedy)
    algo.config = dataclasses.replace(algo.config, epsilon_start=1.0, epsilon_end=1.0)
    random_actions = algo.act(obs)
    assert abs(random_actions.mean() - 0.5) < 0.03
    assert 0.4 < (random_actions != greedy).mean() < 0.6


# ----------------------------------------------------------------------
# Targets, mixers and losses
# ----------------------------------------------------------------------
def test_vdn_mixes_by_summation():
    envs = linemsg()
    algo = VDN(envs, seed=0)
    envs.close()
    q = torch.randn(5, 4)
    obs = torch.randn(5, 4, 3)
    torch.testing.assert_close(algo.mix(q, obs), q.sum(-1))
    torch.testing.assert_close(algo.mix(q, obs, target=True), q.sum(-1))
    assert list(algo.mixer.parameters()) == []


def test_qmix_mixer_is_monotonic():
    envs = wireless()
    algo = QMIX(envs, seed=0, mixer_embed_dim=8, hypernet_hidden=16)
    envs.close()
    perturb(algo.mixer, scale=1.0)  # arbitrary weights: monotonicity must hold for all of them
    generator = torch.Generator().manual_seed(0)
    q = (5.0 * torch.randn(64, 16, generator=generator)).requires_grad_(True)
    obs = torch.rand(64, 16, 18, generator=generator)
    q_tot = algo.mix(q, obs)
    assert q_tot.shape == (64,)
    (grad,) = torch.autograd.grad(q_tot.sum(), q)  # sample b only depends on q[b]
    assert torch.all(grad >= 0.0)
    assert torch.any(grad > 0.0)
    # Increasing any single utility never decreases the team value.
    bumped = q.detach().clone()
    bumped[:, 3] += 1.0
    assert torch.all(algo.mix(bumped, obs) >= q_tot.detach() - 1e-5)


@pytest.mark.parametrize("cls", ALGORITHMS)
@pytest.mark.parametrize("double_q", [True, False])
def test_td_target_on_hand_made_batch(cls, double_q):
    envs = linemsg()
    algo = cls(envs, seed=0, double_q=double_q, gamma=0.9, reward_scale=2.0)
    envs.close()
    perturb(algo.target_q_net)  # online and target networks now disagree
    batch = hand_batch(algo)
    tensors = as_tensors(algo, batch)

    with torch.no_grad():
        q_online = algo.q_values(tensors["next_obs"]).numpy()
        q_target = algo.q_values(tensors["next_obs"], target=True).numpy()
    size, n = batch["actions"].shape
    selected = np.zeros((size, n), dtype=np.float32)
    for b in range(size):
        for i in range(n):
            chooser = q_online if double_q else q_target
            selected[b, i] = q_target[b, i, np.argmax(chooser[b, i])]
    not_terminated = 1.0 - batch["terminated"]
    if cls is IQL:
        expected = 2.0 * batch["agent_rewards"] + 0.9 * not_terminated[:, None] * selected
    else:
        with torch.no_grad():
            mixed = algo.mix(torch.as_tensor(selected), tensors["next_obs"], target=True).numpy()
        if cls is VDN:
            np.testing.assert_allclose(mixed, selected.sum(1), rtol=1e-5, atol=1e-5)
        expected = 2.0 * batch["reward"] + 0.9 * not_terminated * mixed

    target = algo.td_target(tensors)
    assert target.shape == expected.shape
    np.testing.assert_allclose(target.numpy(), expected, rtol=1e-5, atol=1e-5)
    # Terminal transitions do not bootstrap.
    rewards = batch["agent_rewards"] if cls is IQL else batch["reward"]
    terminal = batch["terminated"] == 1.0
    np.testing.assert_allclose(target.numpy()[terminal], 2.0 * rewards[terminal], rtol=1e-6)


def test_double_q_changes_the_target():
    envs = linemsg()
    algo = VDN(envs, seed=0)
    envs.close()
    tensors = as_tensors(algo, hand_batch(algo, size=64))
    not_terminated = 1.0 - tensors["terminated"]

    # Online network prefers action 0, target network action 1 (by 10 units): double Q
    # evaluates the target network at action 0, plain Q-learning takes its maximum.
    with torch.no_grad():
        algo.q_net.nets[0][-1].bias.copy_(torch.tensor([10.0, 0.0]))
        algo.target_q_net.nets[0][-1].bias.copy_(torch.tensor([0.0, 10.0]))
    double = algo.td_target(tensors)
    algo.config = dataclasses.replace(algo.config, double_q=False)
    single = algo.td_target(tensors)
    gap = algo.config.gamma * not_terminated * 4 * 10.0  # 4 agents
    torch.testing.assert_close(single - double, gap, atol=5.0, rtol=0.0)

    # max_a Q^-(o', a) >= Q^-(o', argmax_a Q(o', a)) for every agent, so single >= double.
    perturb(algo.target_q_net)
    single = algo.td_target(tensors)
    algo.config = dataclasses.replace(algo.config, double_q=True)
    double = algo.td_target(tensors)
    assert torch.all(single >= double - 1e-5)


@pytest.mark.parametrize("loss", ["huber", "mse"])
def test_td_loss_matches_definition(loss):
    envs = linemsg()
    algo = QMIX(envs, seed=0, loss=loss)
    envs.close()
    tensors = as_tensors(algo, hand_batch(algo))
    value, stats = algo.td_loss(tensors)
    with torch.no_grad():
        chosen = algo.q_values(tensors["obs"]).gather(-1, tensors["actions"][..., None])[..., 0]
        q_tot = algo.mix(chosen, tensors["obs"])
        error = q_tot - algo.td_target(tensors)
    if loss == "mse":
        expected = error.pow(2).mean()
    else:
        expected = torch.where(error.abs() < 1, 0.5 * error.pow(2), error.abs() - 0.5).mean()
    torch.testing.assert_close(value.detach(), expected)
    assert stats["td_loss"] == pytest.approx(float(expected), rel=1e-5)
    assert stats["q_mean"] == pytest.approx(float(q_tot.mean()), rel=1e-5)


@pytest.mark.parametrize("interval", [0, 3])
def test_target_updates(interval):
    envs = linemsg()
    algo = QMIX(envs, seed=0, tau=0.1, target_update_interval=interval)
    envs.close()
    batch = hand_batch(algo, size=16)
    initial = [p.detach().clone() for p in algo.target_mixer.parameters()]
    for step in range(3):
        before = [p.detach().clone() for p in algo.target_q_net.parameters()]
        algo._update(batch)
        algo.num_updates += 1  # done by the training loop
        online = list(algo.q_net.parameters())
        after = list(algo.target_q_net.parameters())
        if interval == 0:  # Polyak averaging after every step
            for b, a, o in zip(before, after, online):
                torch.testing.assert_close(a, 0.9 * b + 0.1 * o)
        elif step < 2:  # hard copy only every third step
            for b, a in zip(before, after):
                torch.testing.assert_close(a, b)
        else:
            for a, o in zip(after, online):
                torch.testing.assert_close(a, o)
    changed = [not torch.equal(i, p) for i, p in zip(initial, algo.target_mixer.parameters())]
    assert any(changed)


# ----------------------------------------------------------------------
# Training, determinism, persistence
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", ALGORITHMS)
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "unshared"])
def test_short_learn(cls, shared):
    envs = wireless()
    algo = cls(envs, seed=0, share_parameters=shared, **FAST)
    log = algo.learn(envs, 320, seed=0)
    envs.close()
    assert algo.env_steps == 320
    assert len(algo.q_net.nets) == (1 if shared else 16)
    assert algo.agent_input_dim == 18 + (16 if shared else 0)
    assert log.updates and algo.num_updates == len(log.updates)
    stats = log.updates[-1][1]
    assert {"td_loss", "q_mean", "target_mean", "grad_norm", "epsilon"} <= set(stats)
    assert all(np.isfinite(v) for v in stats.values())
    assert stats["epsilon"] == pytest.approx(algo.epsilon_at(320))
    assert algo.act(np.zeros((2, 16, 18), np.float32)).shape == (2, 16)


@pytest.mark.parametrize("cls", ALGORITHMS)
def test_seed_determinism(cls):
    def run(seed):
        envs = linemsg()
        algo = cls(envs, seed=seed, **FAST)
        log = algo.learn(envs, 240, seed=seed)
        envs.close()
        return algo, log

    (a, log_a), (b, log_b), (c, _) = run(3), run(3), run(4)
    for name, module in a._modules().items():
        if isinstance(module, torch.nn.Module):
            for pa, pb in zip(module.parameters(), b._modules()[name].parameters()):
                assert torch.equal(pa, pb)
    assert log_a.episodes == log_b.episodes
    assert log_a.updates == log_b.updates
    assert not all(
        torch.equal(pa, pc) for pa, pc in zip(a.q_net.parameters(), c.q_net.parameters())
    )


@pytest.mark.parametrize("cls", ALGORITHMS)
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "unshared"])
def test_save_load_round_trip(cls, shared, tmp_path):
    envs = linemsg()
    algo = cls(envs, seed=0, share_parameters=shared, **FAST)
    algo.learn(envs, 160, seed=0)
    envs.close()
    path = algo.save(tmp_path / f"{cls.name}.pt")
    for loader in (cls, Algorithm):
        loaded = loader.load(path)
        assert type(loaded) is cls and loaded.config == algo.config
        assert loaded.env_steps == algo.env_steps and loaded.num_updates == algo.num_updates
        obs = np.random.default_rng(1).random((64, 4, 3)).astype(np.float32)
        np.testing.assert_array_equal(
            loaded.act(obs, deterministic=True), algo.act(obs, deterministic=True)
        )
        tensors = as_tensors(algo, hand_batch(algo))
        torch.testing.assert_close(loaded.td_target(tensors), algo.td_target(tensors))


# ----------------------------------------------------------------------
# Learning sanity and presets
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", [VDN, QMIX])
def test_learns_to_relay_on_linemsg(cls):
    envs = make_vector_env("LineMsg-v0", 8, num_agents=4)
    algo = cls(
        envs,
        seed=0,
        hidden_sizes=(32, 32),
        batch_size=64,
        warmup_steps=200,
        epsilon_decay_steps=1500,
        lr=2e-3,
        gamma=0.9,
    )
    algo.learn(envs, 2400, seed=0)
    envs.close()
    eval_envs = make_vector_env("LineMsg-v0", 16, num_agents=4)
    trained = algo.evaluate(eval_envs, 16, seed=1).mean_return
    random = env_lib.evaluate(eval_envs, None, n_episodes=16, seed=1).mean_return
    eval_envs.close()
    # Always relaying earns 50 * (1 + 0.3) = 65; uniformly random actions about 29.
    assert trained > 0.9 * 65.0 > random + 20.0


def test_presets_are_valid():
    assert set(PRESETS) == {"iql", "vdn", "qmix"}
    registered = set(env_lib.list_envs())
    for algorithm, table in PRESETS.items():
        assert table, f"no presets for {algorithm}"
        for env_id, preset in table.items():
            assert env_id in registered
            assert {"total_steps", "num_envs", "config"} <= set(preset)
            QLearningConfig(**preset["config"])  # valid overrides


def test_train_entry_point():
    algo, log = train("iql", "LineMsg-v0", 128, num_envs=4, env_kwargs={"num_agents": 3}, **FAST)
    assert isinstance(algo, IQL) and log.env_id == "LineMsg-v0" and algo.env_steps == 128
