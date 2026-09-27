"""Tests for the off-policy actor-critic family: MADDPG and MATD3."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import env_lib
from marl_algorithms import MADDPG, MATD3, DDPGConfig, get_algorithm, make_vector_env, train
from marl_algorithms.algorithms.ddpg import PRESETS

# Small continuous environments: (env_id, kwargs, n_agents, obs_dim, action_dim).
CONTINUOUS = [
    ("PowerGrid-v0", {"n_buses": 6}, 6, 10, 1),
    ("Consensus-v0", {"n_agents": 3}, 3, 16, 2),
    ("Platoon-v0", {"n_followers": 3}, 3, 13, 1),
]
FAST = {"batch_size": 32, "warmup_steps": 64, "hidden_sizes": (16, 16)}


def _params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(b, p) for b, p in zip(before, module.parameters()))


def _filled(cls, env_id="PowerGrid-v0", env_kwargs=None, steps=256, **config):
    """An algorithm whose replay buffer holds ``steps`` random transitions (no update yet)."""
    envs = make_vector_env(env_id, 8, **(env_kwargs or {"n_buses": 6}))
    algo = cls(envs, seed=0, **{**FAST, "warmup_steps": 10**6, **config})
    algo.learn(envs, steps, seed=0)
    assert algo.num_updates == 0
    envs.close()
    return algo


# ---------------------------------------------------------------------------
# Construction and acting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [MADDPG, MATD3])
@pytest.mark.parametrize(("env_id", "kwargs", "n", "obs_dim", "action_dim"), CONTINUOUS)
def test_construction_and_action_shapes(cls, env_id, kwargs, n, obs_dim, action_dim):
    envs = make_vector_env(env_id, 4, **kwargs)
    algo = cls(envs, seed=0, **FAST)
    spec = algo.spec
    assert (spec.n_agents, spec.obs_dim, spec.action_dim) == (n, obs_dim, action_dim)
    obs, _ = envs.reset(seed=0)
    obs = spec.agent_obs(obs)
    for deterministic in (True, False):
        actions = algo.act(obs, deterministic=deterministic)
        assert actions.shape == (4, n, action_dim) and actions.dtype == np.float32
        assert np.all(actions >= spec.action_low) and np.all(actions <= spec.action_high)
    # Shared critic: global state, joint action, own observation and action, agent id.
    obs_t = algo.tensor(obs)
    actions = torch.rand(4, n, action_dim)
    inputs = algo._critic_inputs(obs_t, actions)
    state_dim, joint_dim = n * obs_dim, n * action_dim
    assert inputs.shape == (4, n, state_dim + joint_dim + obs_dim + action_dim + n)
    assert algo.critic(inputs).shape == (4, n, algo.n_critics)
    for i in range(n):
        row = inputs[:, i]
        torch.testing.assert_close(row[:, :state_dim], obs_t.flatten(1))
        torch.testing.assert_close(row[:, state_dim : state_dim + joint_dim], actions.flatten(1))
        torch.testing.assert_close(row[:, state_dim + joint_dim :][:, :obs_dim], obs_t[:, i])
        torch.testing.assert_close(row[:, -n - action_dim : -n], actions[:, i])
        torch.testing.assert_close(row[:, -n:], torch.eye(n)[i].expand(4, n))
    plain = cls(envs, seed=0, critic_local_inputs=False, share_parameters=False, **FAST)
    assert plain._critic_inputs(obs_t, actions).shape == (4, n, state_dim + joint_dim)
    envs.close()


@pytest.mark.parametrize("cls", [MADDPG, MATD3])
def test_policy_runs_through_env_lib_evaluate(cls):
    env = env_lib.make("Consensus-v0", n_agents=3, max_steps=10)
    algo = cls(env, seed=0, **FAST)
    result = env_lib.evaluate(env, algo.policy(), n_episodes=2, seed=0)
    assert result.n_episodes == 2 and np.all(result.lengths <= 10)
    envs = env_lib.make_vec("Consensus-v0", 4, n_agents=3, max_steps=10)
    assert algo.evaluate(envs, n_episodes=4, seed=0).n_episodes == 4


@pytest.mark.parametrize("cls", [MADDPG, MATD3])
def test_discrete_environment_raises(cls):
    envs = make_vector_env("LineMsg-v0", 2)
    with pytest.raises(TypeError, match="continuous"):
        cls(envs)
    envs.close()


def test_registry_and_config_validation():
    assert get_algorithm("maddpg") is MADDPG and get_algorithm("matd3") is MATD3
    assert MADDPG.config_class is DDPGConfig and MATD3.n_critics == 2
    with pytest.raises(ValueError, match="reward_source"):
        DDPGConfig(reward_source="global")
    with pytest.raises(ValueError, match="policy_delay"):
        DDPGConfig(policy_delay=0)
    with pytest.raises(ValueError, match="critic_hidden_sizes"):
        DDPGConfig(critic_hidden_sizes=())
    with pytest.raises(ValueError, match="critic_loss"):
        DDPGConfig(critic_loss="l1")
    with pytest.raises(TypeError, match="unknown"):
        MADDPG(env_lib.make("Consensus-v0", n_agents=3), no_such_option=1)


def test_heterogeneous_action_bounds_are_respected():
    u_max = np.array([0.05, 0.1, 0.2, 0.1, 0.1, 0.3])
    env = env_lib.make("PowerGrid-v0", n_buses=6, u_max=u_max)
    algo = MADDPG(env, seed=0, exploration_noise=1.0, **FAST)
    obs = algo.spec.agent_obs(env.reset(seed=0)[0])
    actions = algo.act(np.repeat(obs[None], 64, axis=0), deterministic=False)[..., 0]
    bound = u_max.astype(np.float32)
    assert np.all(np.abs(actions) <= bound)
    np.testing.assert_allclose(np.abs(actions).max(axis=0), bound)  # clipped at each bound


def test_exploration_noise_decays_linearly():
    env = env_lib.make("Consensus-v0", n_agents=3)
    algo = MADDPG(
        env, exploration_noise=0.4, final_exploration_noise=0.1, noise_decay_steps=1000, **FAST
    )
    assert algo.exploration_scale == pytest.approx(0.4)
    algo.env_steps = 500
    assert algo.exploration_scale == pytest.approx(0.25)
    algo.env_steps = 5000
    assert algo.exploration_scale == pytest.approx(0.1)
    constant = MADDPG(env, exploration_noise=0.3, **FAST)
    constant.env_steps = 10_000
    assert constant.exploration_scale == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [MADDPG, MATD3])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("reward_source", ["agent", "team"])
def test_short_learn_runs(cls, shared, reward_source):
    envs = make_vector_env("Consensus-v0", 8, n_agents=3, max_steps=20)
    algo = cls(envs, seed=0, share_parameters=shared, reward_source=reward_source, **FAST)
    log = algo.learn(envs, 320, seed=0)
    # Updates start on the vector step that completes the warm-up.
    assert algo.env_steps == 320 and algo.num_updates == (320 - 64) // 8 + 1
    assert len(algo.actor.nets) == len(algo.critic.nets) == (1 if shared else 3)
    assert len(log.episodes) >= 16
    stats = log.updates[-1][1]
    assert set(stats) == {"critic_loss", "actor_loss", "q_mean", "target_mean", "noise_scale"}
    assert all(np.isfinite(v) for v in stats.values())
    envs.close()


def test_actor_critic_inputs_replace_only_the_own_action():
    algo = MADDPG(env_lib.make("Consensus-v0", n_agents=3), **FAST)
    obs = torch.zeros(2, 3, algo.spec.obs_dim)
    replay = torch.arange(12.0).reshape(2, 3, 2)
    own = -torch.arange(12.0).reshape(2, 3, 2) - 1.0
    joint = algo._own_action_joint(own, replay).reshape(2, 3, 3, 2)
    for i in range(3):
        for j in range(3):
            expected = own[:, j] if i == j else replay[:, j]
            assert torch.equal(joint[:, i, j], expected)
    inputs = algo._critic_inputs(obs, replay, own)
    state_dim = 3 * algo.spec.obs_dim
    torch.testing.assert_close(inputs[..., state_dim : state_dim + 6], joint.flatten(-2))
    torch.testing.assert_close(inputs[..., -3 - 2 : -3], own)  # repeated own action


def test_actor_gradient_flows_only_through_own_action():
    algo = MADDPG(env_lib.make("Consensus-v0", n_agents=3), **FAST)
    own = torch.zeros(1, 3, 2, requires_grad=True)
    joint = algo._own_action_joint(own, torch.ones(1, 3, 2))
    # Critic 0 (row 0) depends on agent 0's own action only.
    joint[0, 0].sum().backward()
    assert own.grad[0, 0].abs().sum() > 0 and own.grad[0, 1:].abs().sum() == 0


@pytest.mark.parametrize("reward_source", ["agent", "team"])
def test_td_targets_use_reward_source_scale_and_terminal_mask(reward_source):
    algo = _filled(MADDPG, reward_source=reward_source, gamma=0.9, reward_scale=2.0)
    batch = algo.replay.sample(32, algo.np_rng)
    batch["agent_rewards"] = np.tile(np.arange(6, dtype=np.float32), (32, 1))
    batch["reward"] = np.full(32, 10.0, dtype=np.float32)
    batch["terminated"] = np.ones(32, dtype=np.float32)  # no bootstrapping
    stats = algo._update(batch)
    expected = 2.0 * (2.5 if reward_source == "agent" else 10.0)
    assert stats["target_mean"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("critic_loss", ["mse", "huber"])
def test_critic_loss_matches_its_definition(critic_loss):
    algo = _filled(MADDPG, critic_loss=critic_loss, gamma=0.0, reward_scale=1.0)
    batch = algo.replay.sample(32, algo.np_rng)
    rewards = np.zeros((32, 6), dtype=np.float32)
    rewards[:4] = 50.0  # a few outliers far outside the quadratic zone of the Huber loss
    batch["agent_rewards"] = rewards
    obs, _ = algo._prepare_obs(batch)
    actions = algo.tensor(batch["actions"]) / algo._action_half_range
    with torch.no_grad():
        errors = algo.critic(algo._critic_inputs(obs, actions))[..., 0] - algo.tensor(rewards)
    if critic_loss == "mse":
        expected = errors.pow(2).mean()
    else:
        absolute = errors.abs()
        expected = torch.where(absolute < 1.0, errors.pow(2), 2.0 * absolute - 1.0).mean()
    assert algo._update(batch)["critic_loss"] == pytest.approx(expected.item(), rel=1e-5)


def test_matd3_target_is_the_minimum_of_the_twin_critics():
    algo = _filled(MATD3, gamma=0.5, reward_scale=1.0)
    for k, value in enumerate((3.0, -1.0)):  # constant target critics Q'_1 = 3, Q'_2 = -1
        head = algo.critic_target.nets[0].nets[k][-1]
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.constant_(head.bias, value)
    batch = algo.replay.sample(32, algo.np_rng)
    batch["agent_rewards"] = np.zeros((32, 6), dtype=np.float32)
    batch["terminated"] = np.zeros(32, dtype=np.float32)
    assert algo._update(batch)["target_mean"] == pytest.approx(0.5 * -1.0)


def test_matd3_delays_actor_and_target_updates():
    algo = _filled(MATD3, policy_delay=3)
    for update in range(1, 7):
        actor, critic = _params(algo.actor), _params(algo.critic)
        actor_target, critic_target = _params(algo.actor_target), _params(algo.critic_target)
        algo._update(algo.replay.sample(32, algo.np_rng))
        actor_step = update % 3 == 0
        assert _changed(critic, algo.critic)
        assert _changed(actor, algo.actor) == actor_step
        assert _changed(actor_target, algo.actor_target) == actor_step
        assert _changed(critic_target, algo.critic_target) == actor_step


def test_maddpg_updates_actor_every_step_and_ignores_policy_delay():
    algo = _filled(MADDPG, policy_delay=3)
    for _ in range(2):
        actor, critic_target = _params(algo.actor), _params(algo.critic_target)
        algo._update(algo.replay.sample(32, algo.np_rng))
        assert _changed(actor, algo.actor) and _changed(critic_target, algo.critic_target)


def test_actor_step_does_not_change_the_critic():
    algo = _filled(MADDPG)
    batch = algo.replay.sample(32, algo.np_rng)
    obs, _ = algo._prepare_obs(batch)
    actions = algo.tensor(batch["actions"]) / algo._action_half_range
    critic = _params(algo.critic)
    algo._update_actor(obs, actions)
    assert not _changed(critic, algo.critic)
    assert all(p.requires_grad for p in algo.critic.parameters())


def test_observation_statistics_are_frozen_after_warmup():
    envs = make_vector_env("Consensus-v0", 8, n_agents=3, max_steps=20)
    algo = MADDPG(envs, seed=0, normalize_observations=True, **FAST)
    algo.learn(envs, 56, seed=0)  # warm-up only: no gradient step yet
    assert algo.num_updates == 0 and not algo._obs_stats_frozen
    algo.learn(envs, 256, seed=0)
    assert algo._obs_stats_frozen
    # The first gradient step (at 64 env steps) used the 64 warm-up transitions.
    warmup_obs = algo.replay.data["obs"][:64].reshape(-1, algo.spec.obs_dim)
    np.testing.assert_allclose(
        algo.obs_normalizer.rms.mean, warmup_obs.mean(axis=0), rtol=1e-5, atol=1e-6
    )
    mean = algo.obs_normalizer.rms.mean.copy()
    algo.learn(envs, 512, seed=0)
    np.testing.assert_array_equal(algo.obs_normalizer.rms.mean, mean)
    envs.close()


def test_maddpg_learns_a_tiny_rendezvous_task():
    # Three fully connected agents in a small arena: moving towards the others
    # pays off within a few steps. Zero actions score about -79, uniformly
    # random ones about -92, the Laplacian controller about +0.2.
    kwargs = {"n_agents": 3, "topology": "complete", "arena_size": 2.0, "max_steps": 25}
    envs = make_vector_env("Consensus-v0", 8, **kwargs)
    algo = MADDPG(envs, seed=0, hidden_sizes=(32, 32), batch_size=64, warmup_steps=400, gamma=0.8)
    algo.learn(envs, 3000, seed=0)
    result = algo.evaluate(env_lib.make_vec("Consensus-v0", 16, **kwargs), n_episodes=32, seed=1)
    assert result.mean_return > -60.0
    envs.close()


# ---------------------------------------------------------------------------
# Reproducibility and persistence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [MADDPG, MATD3])
def test_same_seed_gives_identical_training(cls):
    def run():
        envs = make_vector_env("PowerGrid-v0", 8, n_buses=6, max_steps=30)
        algo = cls(envs, seed=3, **FAST)
        log = algo.learn(envs, 256, seed=3)
        envs.close()
        return algo, log

    (a, log_a), (b, log_b) = run(), run()
    for name in ("actor", "critic", "actor_target", "critic_target"):
        state_a, state_b = getattr(a, name).state_dict(), getattr(b, name).state_dict()
        assert all(torch.equal(state_a[key], state_b[key]) for key in state_a)

    def table(log):
        return np.array([[stats[k] for k in sorted(stats)] for _, stats in log.updates])

    np.testing.assert_array_equal(table(log_a), table(log_b))  # NaN-aware


@pytest.mark.parametrize("cls", [MADDPG, MATD3])
@pytest.mark.parametrize("shared", [True, False])
def test_save_load_round_trip(cls, shared, tmp_path):
    envs = make_vector_env("Consensus-v0", 8, n_agents=3, max_steps=20)
    algo = cls(envs, seed=0, share_parameters=shared, normalize_observations=True, **FAST)
    algo.learn(envs, 192, seed=0)
    path = algo.save(tmp_path / "agent.pt")
    loaded = type(algo).load(path)
    assert type(loaded) is cls and loaded.config == algo.config
    assert loaded.env_steps == algo.env_steps and loaded._obs_stats_frozen
    obs = algo.spec.agent_obs(envs.reset(seed=1)[0])
    np.testing.assert_array_equal(
        loaded.act(obs, deterministic=True), algo.act(obs, deterministic=True)
    )
    envs.close()


def test_train_entry_point_on_platoon():
    algo, log = train(
        "matd3",
        "Platoon-v0",
        total_steps=256,
        num_envs=8,
        env_kwargs={"n_followers": 2},
        batch_size=32,
        warmup_steps=128,
        hidden_sizes=(16, 16),
    )
    assert isinstance(algo, MATD3) and log.env_id == "Platoon-v0"
    assert algo.num_updates == 17 and np.isfinite(log.updates[-1][1]["critic_loss"])


def test_presets_are_well_formed():
    assert set(PRESETS) == {"maddpg", "matd3"}
    for name, presets in PRESETS.items():
        assert {"PowerGrid-v0", "Consensus-v0"} <= set(presets)
        for preset in presets.values():
            assert set(preset) == {"num_envs", "total_steps", "config", "env_kwargs"}
            DDPGConfig(**preset["config"])  # valid overrides
            assert get_algorithm(name).name == name
