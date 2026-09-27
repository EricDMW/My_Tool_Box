"""Tests for the shared marl_algorithms core (spec, runner, buffers, normalisation, networks)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import env_lib
from marl_algorithms.core import (
    MultiAgentSpec,
    ObservationNormalizer,
    ReplayBuffer,
    RewardScaler,
    RolloutBuffer,
    RunningMeanStd,
    VectorRunner,
    compute_gae,
    make_vector_env,
)
from marl_algorithms.core.networks import (
    CategoricalPolicy,
    DeterministicPolicy,
    GaussianPolicy,
    PerAgent,
    QMixer,
    mlp,
    soft_update,
)
from marl_algorithms.registry import get_algorithm, list_algorithms


# ---------------------------------------------------------------------------
# MultiAgentSpec
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("env_id", "kwargs", "n", "obs_dim", "kind", "action_dim", "n_actions"),
    [
        ("PowerGrid-v0", {}, 16, 10, "continuous", 1, None),
        ("Consensus-v0", {}, 8, 16, "continuous", 2, None),
        ("LineMsg-v0", {"action_space_type": "multibinary"}, 10, 3, "discrete", 1, 2),
        ("WirelessComm-v1", {}, 16, 18, "discrete", 1, 5),
        ("KuramotoOscillator-v0", {}, 1, 75, "continuous", 55, None),
    ],
)
def test_spec_from_env(env_id, kwargs, n, obs_dim, kind, action_dim, n_actions):
    env = env_lib.make(env_id, **kwargs)
    spec = MultiAgentSpec.from_env(env)
    assert (spec.n_agents, spec.obs_dim, spec.action_kind) == (n, obs_dim, kind)
    assert (spec.action_dim, spec.n_actions) == (action_dim, n_actions)
    assert spec.state_dim == n * obs_dim
    obs, _ = env.reset(seed=0)
    assert spec.agent_obs(obs).shape == (n, obs_dim)
    assert spec.agent_obs(np.stack([obs, obs])).shape == (2, n, obs_dim)
    if spec.continuous:
        actions = np.zeros((n, action_dim), dtype=np.float32)
    else:
        actions = np.zeros(n, dtype=np.int64)
    assert env.action_space.contains(spec.env_action(actions))
    env.close()


def test_spec_clips_continuous_actions_and_accepts_squeezed_shape():
    env = env_lib.make("PowerGrid-v0")
    spec = MultiAgentSpec.from_env(env)
    joint = spec.env_action(np.full((16,), 5.0))
    assert joint.shape == (16,) and np.allclose(joint, spec.action_high[:, 0])
    batch = spec.env_action(np.full((3, 16, 1), -5.0))
    assert batch.shape == (3, 16) and np.allclose(batch, spec.action_low[:, 0])
    env.close()


def test_spec_rejects_joint_discrete_action():
    env = env_lib.make("LineMsg-v0")  # Discrete(2**N) joint action
    with pytest.raises(TypeError, match="multibinary"):
        MultiAgentSpec.from_env(env)
    env.close()


def test_spec_from_vector_env_uses_single_spaces():
    envs = make_vector_env("Consensus-v0", 3)
    spec = MultiAgentSpec.from_env(envs)
    assert spec.n_agents == 8 and spec.action_dim == 2
    envs.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def test_make_vector_env_uses_native_batch_and_same_step():
    envs = make_vector_env("PowerGrid-v0", 4)
    assert type(envs).__name__ == "PowerGridVectorEnv"
    assert envs.autoreset_mode == "same_step"
    envs.close()
    envs = make_vector_env("LineMsg-v0", 2)
    assert MultiAgentSpec.from_env(envs).n_actions == 2
    envs.close()


def test_runner_requires_same_step_autoreset():
    envs = env_lib.make_vec("Consensus-v0", 2)
    with pytest.raises(ValueError, match="same_step"):
        VectorRunner(envs)
    envs.close()


def test_runner_transitions_and_episode_ends():
    envs = make_vector_env("Consensus-v0", 3, max_steps=4)
    runner = VectorRunner(envs)
    obs = runner.reset(seed=0)
    assert obs.shape == (3, 8, 16)
    zero = np.zeros((3, 8, 2), dtype=np.float32)
    for _ in range(4):
        before = runner.obs
        transition = runner.step(zero)
        assert np.array_equal(transition.obs, before)
        assert transition.agent_rewards.shape == (3, 8)
        # Consensus reports per-agent rewards whose mean is the team reward.
        np.testing.assert_allclose(
            transition.agent_rewards.mean(axis=1), transition.reward, rtol=1e-5
        )
    assert transition.truncated.all() and not transition.terminated.any()
    # next_obs of finished copies is the final observation, obs of the runner the reset one.
    assert not np.array_equal(transition.next_obs, runner.obs)
    episodes = runner.pop_episodes()
    assert len(episodes) == 3 and all(length == 4 for _, length in episodes)
    assert runner.pop_episodes() == []
    envs.close()


def test_runner_on_sync_vector_env_with_per_agent_rewards():
    envs = make_vector_env("AJLATT-v0", 2, max_episode_steps=2)
    runner = VectorRunner(envs)
    runner.reset(seed=0)
    spec = runner.spec
    actions = np.zeros((2, spec.n_agents, spec.action_dim), dtype=np.float32)
    runner.step(actions)
    transition = runner.step(actions)
    assert transition.truncated.all()
    assert transition.agent_rewards.shape == (2, 4)
    assert np.all(np.isfinite(transition.agent_rewards))
    envs.close()


# ---------------------------------------------------------------------------
# Buffers
# ---------------------------------------------------------------------------
def test_rollout_buffer_fill_and_reset():
    buf = RolloutBuffer(3, 2, {"x": ((4,), np.float32), "flag": ((), bool)})
    for t in range(3):
        buf.add(x=np.full((2, 4), t), flag=np.array([t % 2 == 0, False]))
    assert buf.full and buf["x"].shape == (3, 2, 4) and "flag" in buf
    with pytest.raises(RuntimeError):
        buf.add(x=np.zeros((2, 4)), flag=np.zeros(2, bool))
    buf.reset()
    assert buf["x"].shape == (0, 2, 4)
    with pytest.raises(KeyError):
        buf.add(x=np.zeros((2, 4)))


def test_compute_gae_matches_a_manual_recursion():
    rng = np.random.default_rng(0)
    T, B, n = 6, 3, 2
    rewards = rng.normal(size=(T, B, n))
    values = rng.normal(size=(T, B, n))
    next_values = rng.normal(size=(T, B, n))
    terminated = rng.random((T, B)) < 0.2
    truncated = (rng.random((T, B)) < 0.2) & ~terminated
    gamma, lam = 0.9, 0.8
    adv, ret = compute_gae(rewards, values, next_values, terminated, truncated, gamma, lam)
    expected = np.zeros_like(rewards)
    for b in range(B):
        running = np.zeros(n)
        for t in reversed(range(T)):
            delta = (
                rewards[t, b] + gamma * (not terminated[t, b]) * next_values[t, b] - values[t, b]
            )
            done = terminated[t, b] or truncated[t, b]
            running = delta + gamma * lam * (not done) * running
            expected[t, b] = running
    np.testing.assert_allclose(adv, expected)
    np.testing.assert_allclose(ret, expected + values)


@pytest.mark.parametrize("env_id", ["PowerGrid-v0", "WirelessComm-v1"])
def test_replay_buffer_wraps_and_samples(env_id):
    envs = make_vector_env(env_id, 4)
    runner = VectorRunner(envs)
    runner.reset(seed=0)
    spec = runner.spec
    replay = ReplayBuffer(10, spec)
    rng = np.random.default_rng(0)
    for _ in range(4):
        if spec.continuous:
            actions = np.zeros((4, spec.n_agents, spec.action_dim), dtype=np.float32)
        else:
            actions = rng.integers(0, spec.n_actions, (4, spec.n_agents))
        replay.add(runner.step(actions))
    assert len(replay) == 10 and replay.pos == 6
    batch = replay.sample(5, rng)
    assert batch["obs"].shape == (5, spec.n_agents, spec.obs_dim)
    assert batch["agent_rewards"].shape == (5, spec.n_agents)
    assert batch["actions"].dtype == (np.float32 if spec.continuous else np.int64)
    envs.close()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(1)
    data = rng.normal(3.0, 2.0, size=(500, 4))
    rms = RunningMeanStd((4,))
    for chunk in np.array_split(data, 7):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, data.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(rms.var, data.var(axis=0), rtol=1e-3)
    clone = RunningMeanStd((4,))
    clone.load_state_dict(rms.state_dict())
    np.testing.assert_array_equal(clone.mean, rms.mean)


def test_observation_normalizer_and_reward_scaler():
    norm = ObservationNormalizer(3, clip=5.0)
    norm.update(np.random.default_rng(0).normal(10.0, 4.0, size=(1000, 3)))
    out = norm(np.full((2, 3), 10.0))
    assert out.dtype == np.float32 and np.all(np.abs(out) < 0.2)
    assert np.all(np.abs(norm(np.full((1, 3), 1e6))) <= 5.0)
    off = ObservationNormalizer(3, enabled=False)
    assert np.array_equal(off(np.ones((2, 3))), np.ones((2, 3), dtype=np.float32))
    scaler = RewardScaler(2, gamma=0.9)
    for _ in range(50):
        scaled = scaler(np.full((2, 4), 100.0), np.zeros(2, bool))
    assert np.all(np.abs(scaled) < 100.0)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def test_mlp_and_invalid_activation():
    net = mlp(4, 3, (8, 8), "relu", layer_norm=True)
    assert net(torch.zeros(5, 4)).shape == (5, 3)
    with pytest.raises(ValueError):
        mlp(4, 3, activation="sigmoid")


def test_policies_shapes_and_log_probs():
    gen = torch.Generator().manual_seed(0)
    low, high = np.array([-1.0, 0.0]), np.array([1.0, 4.0])
    gaussian = GaussianPolicy(5, 2, low, high)
    x = torch.zeros(3, 4, 5)
    action, logp = gaussian.sample(x, gen)
    assert action.shape == (3, 4, 2) and logp.shape == (3, 4)
    logp2, entropy = gaussian.log_prob(x, action)
    torch.testing.assert_close(logp, logp2)
    assert entropy.shape == (3, 4)
    mean, _ = gaussian(x)
    torch.testing.assert_close(mean, torch.tensor([0.0, 2.0]).expand(3, 4, 2), atol=0.1, rtol=0)

    categorical = CategoricalPolicy(5, 3)
    action, logp = categorical.sample(x, gen)
    assert action.shape == (3, 4) and action.dtype == torch.int64
    logp2, entropy = categorical.log_prob(x, action)
    torch.testing.assert_close(logp, logp2)
    assert torch.all(entropy <= np.log(3) + 1e-6)

    deterministic = DeterministicPolicy(5, 2, low, high)
    out = deterministic(torch.randn(10, 5, generator=gen) * 100)
    assert torch.all(out >= torch.tensor(low, dtype=torch.float32) - 1e-6)
    assert torch.all(out <= torch.tensor(high, dtype=torch.float32) + 1e-6)


def test_per_agent_shared_and_separate():
    shared = PerAgent(lambda: mlp(5, 2), 3, shared=True)
    separate = PerAgent(lambda: mlp(5, 2), 3, shared=False)
    x = torch.randn(4, 3, 5)
    assert shared(x).shape == separate(x).shape == (4, 3, 2)
    assert len(shared.nets) == 1 and len(separate.nets) == 3
    # Agent i of the separate network only depends on agent i's input.
    y = x.clone()
    y[:, 1] += 1.0
    diff = (separate(y) - separate(x)).abs().sum(dim=(0, 2))
    assert diff[0] == 0 and diff[2] == 0 and diff[1] > 0
    gen = torch.Generator().manual_seed(0)
    low, high = np.zeros(1), np.ones(1)
    policies = PerAgent(lambda: GaussianPolicy(5, 1, low, high), 3, shared=False)
    action, logp = policies.call("sample", x, gen)
    assert action.shape == (4, 3, 1) and logp.shape == (4, 3)
    logp2, _ = policies.call("log_prob", x, action)
    torch.testing.assert_close(logp, logp2)


def test_qmixer_is_monotonic_and_soft_update():
    mixer = QMixer(4, 12)
    q = torch.randn(6, 4, requires_grad=True)
    state = torch.randn(6, 12)
    total = mixer(q, state)
    assert total.shape == (6,)
    (grad,) = torch.autograd.grad(total.sum(), q)
    assert torch.all(grad >= 0)
    a, b = mlp(3, 2), mlp(3, 2)
    soft_update(a, b, 1.0)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_lists_the_classical_algorithms():
    names = [info.name for info in list_algorithms()]
    assert names == ["ippo", "mappo", "maddpg", "matd3", "iql", "vdn", "qmix"]
    with pytest.raises(KeyError):
        get_algorithm("reinforce")


def test_missing_torch_gives_an_install_hint():
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import marl_algorithms\n"
        "try:\n"
        "    marl_algorithms.MAPPO\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert "requires PyTorch" in out and 'pip install "my-tool-box[torch]"' in out


# ---------------------------------------------------------------------------
# Training loop and log
# ---------------------------------------------------------------------------
def test_spec_describe_uses_singular_for_one_agent():
    single = MultiAgentSpec.from_env(env_lib.make("KuramotoOscillator-v0"))
    assert single.describe().startswith("1 agent,")
    team = MultiAgentSpec.from_env(env_lib.make("LineMsg-v0", action_space_type="multibinary"))
    assert team.describe().startswith("10 agents,")


@pytest.mark.parametrize("name", ["mappo", "vdn"])
def test_learn_is_cumulative_and_summary_reports_steps(name):
    envs = make_vector_env("LineMsg-v0", 4, max_iter=10)
    algo = get_algorithm(name)(envs, seed=0, hidden_sizes=(8,), **_SMALL[name])
    log = algo.learn(envs, 256, seed=0)
    first = algo.env_steps
    assert first >= 256
    algo.learn(envs, 2 * first, seed=1, log=log)
    assert 2 * first <= algo.env_steps < 2 * first + 256
    assert f"{algo.env_steps} env steps" in log.summary()
    envs.close()


def test_off_policy_callback_sees_the_episodes_of_its_step():
    envs = make_vector_env("LineMsg-v0", 4, max_iter=10)
    algo = get_algorithm("iql")(envs, seed=0, hidden_sizes=(8,), **_SMALL["vdn"])
    seen = []

    def callback(algorithm, log):
        seen.append((algorithm.env_steps, log.episodes[-1][0] if log.episodes else None))
        return False

    # LineMsg episodes last 10 steps: warm-up and batch size let the first
    # update happen exactly when the first episodes end.
    log = algo.learn(envs, 10_000, seed=0, callback=callback)
    assert len(seen) == 1 and seen[0][0] == seen[0][1] == 40
    assert len(log.episodes) == 4
    envs.close()


_SMALL = {
    "mappo": {"rollout_length": 16, "n_epochs": 1},
    "vdn": {"warmup_steps": 40, "batch_size": 16, "update_every": 1},
}
