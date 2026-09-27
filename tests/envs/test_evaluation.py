"""Tests for env_lib.utils.evaluation (evaluate, rollout, EvaluationResult, Trajectory)."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

import env_lib
from env_lib.utils import EvaluationResult, Trajectory, evaluate, rollout
from env_lib.utils.vector import BatchedVectorEnv


class CountdownEnv(gym.Env):
    """Two agents; every step pays team reward 1 and agent rewards (1, 2).

    The episode ends after ``length`` steps, by termination when ``terminal``
    and by truncation otherwise.
    """

    metadata = {"render_modes": []}

    def __init__(self, length: int = 3, terminal: bool = True):
        self.length = length
        self.terminal = terminal
        self.observation_space = spaces.Box(0.0, np.inf, (2, 1), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2, 1), np.float32)
        self.t = 0

    def _obs(self):
        return np.full((2, 1), self.t, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {"agent_rewards": np.zeros(2)}

    def step(self, action):
        self.t += 1
        done = self.t >= self.length
        info = {"agent_rewards": np.array([1.0, 2.0]), "success": done}
        return self._obs(), 1.0, done and self.terminal, done and not self.terminal, info


class CountdownVectorEnv(BatchedVectorEnv):
    """Native batched version: copy ``i`` ends after ``lengths[i]`` steps."""

    def __init__(self, lengths, **kwargs):
        super().__init__(
            len(lengths),
            spaces.Box(0.0, np.inf, (2, 1), np.float32),
            spaces.Box(-1.0, 1.0, (2, 1), np.float32),
            **kwargs,
        )
        self.lengths = np.asarray(lengths)
        self.t = np.zeros(self.num_envs, dtype=int)

    def _reset_envs(self, mask, options):
        self.t[mask] = 0

    def _step_envs(self, actions, active):
        self.t += 1
        done = self.t >= self.lengths
        infos = {"agent_rewards": np.tile([1.0, 2.0], (self.num_envs, 1)), "success": done.copy()}
        return np.ones(self.num_envs), done, np.zeros(self.num_envs, bool), infos

    def _observe(self):
        return np.repeat(self.t[:, None, None], 2, axis=1).astype(np.float32)


def test_single_env_statistics():
    result = evaluate(CountdownEnv(length=4), n_episodes=3, seed=0)
    assert isinstance(result, EvaluationResult)
    assert result.n_episodes == 3
    np.testing.assert_array_equal(result.returns, [4.0, 4.0, 4.0])
    np.testing.assert_array_equal(result.lengths, [4, 4, 4])
    assert result.termination_rate == 1.0 and result.success_rate == 1.0
    np.testing.assert_array_equal(result.agent_returns, [[4.0, 8.0]] * 3)
    assert result.std_return == 0.0 and result.ci95 == 0.0
    assert result.returns.flags.writeable is False


def test_truncation_and_max_steps():
    result = evaluate(CountdownEnv(length=5, terminal=False), n_episodes=2)
    assert result.termination_rate == 0.0
    capped = evaluate(CountdownEnv(length=50), n_episodes=2, max_steps=7)
    np.testing.assert_array_equal(capped.lengths, [7, 7])
    assert capped.termination_rate == 0.0
    with pytest.raises(ValueError):
        evaluate(CountdownEnv(), n_episodes=0)
    with pytest.raises(ValueError):
        evaluate(CountdownEnv(), max_steps=0)


def test_confidence_interval_matches_student_t():
    stats = pytest.importorskip("scipy.stats")
    returns = np.array([1.0, 2.0, 4.0, 7.0])
    result = EvaluationResult(returns, np.ones(4), np.zeros(4, bool))
    expected = stats.t.ppf(0.975, 3) * returns.std(ddof=1) / 2.0
    assert result.ci95 == pytest.approx(expected)
    assert result.std_return == pytest.approx(returns.std(ddof=1))
    single = EvaluationResult([3.0], [1], [True])
    assert math.isnan(single.ci95) and "n/a" in str(single)


def test_str_and_summary():
    result = evaluate(CountdownEnv(length=2), n_episodes=2, seed=1)
    text = str(result)
    assert "return" in text and "length" in text and "success" in text and "agent return" in text
    assert text.isascii()
    summary = result.summary()
    assert summary["mean_return"] == 2.0 and summary["mean_agent_returns"] == [2.0, 4.0]


def _sync(lengths, mode):
    fns = [lambda n=n: CountdownEnv(length=n) for n in lengths]
    return SyncVectorEnv(fns, autoreset_mode=mode)


@pytest.mark.parametrize(
    "mode", [AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP, AutoresetMode.DISABLED]
)
def test_sync_vector_env_counts_episodes_exactly(mode):
    envs = _sync([2, 3, 5], mode)
    result = evaluate(envs, n_episodes=6, seed=0)
    # Every copy contributes two complete episodes; reset steps are not counted.
    assert sorted(result.returns.tolist()) == [2.0, 2.0, 3.0, 3.0, 5.0, 5.0]
    assert sorted(result.lengths.tolist()) == [2, 2, 3, 3, 5, 5]
    np.testing.assert_array_equal(
        np.sort(result.agent_returns[:, 1]), 2.0 * np.array([2, 2, 3, 3, 5, 5])
    )
    assert result.success_rate == 1.0 and result.termination_rate == 1.0
    envs.close()


@pytest.mark.parametrize("mode", ["next_step", "same_step", "disabled"])
def test_native_vector_env_counts_episodes_exactly(mode):
    envs = CountdownVectorEnv([1, 4], autoreset_mode=mode)
    result = evaluate(envs, n_episodes=5, seed=0)
    # Copy 0 contributes (5 + 0) // 2 = 2 episodes, copy 1 (5 + 1) // 2 = 3.
    assert sorted(result.lengths.tolist()) == [1, 1, 4, 4, 4]
    assert sorted(result.returns.tolist()) == [1.0, 1.0, 4.0, 4.0, 4.0]
    assert sorted(result.agent_returns[:, 0].tolist()) == [1.0, 1.0, 4.0, 4.0, 4.0]
    assert result.success_rate == 1.0


def test_vector_env_max_steps_resets_capped_copies():
    envs = CountdownVectorEnv([2, 100])
    result = evaluate(envs, n_episodes=4, max_steps=3)
    assert sorted(result.lengths.tolist()) == [2, 2, 3, 3]
    assert result.termination_rate == 0.5


def test_seeded_evaluation_is_reproducible():
    env = env_lib.make("Consensus-v0", max_steps=20)
    first = evaluate(env, n_episodes=2, seed=3)
    second = evaluate(env, n_episodes=2, seed=3)
    np.testing.assert_array_equal(first.returns, second.returns)
    other = evaluate(env, n_episodes=2, seed=4)
    assert not np.array_equal(first.returns, other.returns)
    loose = evaluate(env, n_episodes=2, seed=3, deterministic_seeds=False)
    assert loose.returns[0] == first.returns[0]


def test_baseline_evaluation_on_single_and_vector_env():
    env = env_lib.make("Consensus-v0")
    policy = env_lib.baseline_policy(env)
    single = evaluate(env, policy, n_episodes=3, seed=0)
    assert single.policy == "baseline:consensus" and single.env_id == "Consensus-v0"
    envs = env_lib.make_vec("Consensus-v0", num_envs=3, vectorization_mode="sync")
    vector = evaluate(envs, env_lib.baseline_policy(envs), n_episodes=3, seed=0)
    # SyncVectorEnv seeds copy i with seed + i, like deterministic single-env seeds.
    np.testing.assert_allclose(np.sort(vector.returns), np.sort(single.returns))
    assert vector.success_rate == 1.0
    envs.close()


def test_rollout_shapes_and_episode_ids():
    env = CountdownEnv(length=3)
    trajectory = rollout(env, n_episodes=2, seed=0)
    assert isinstance(trajectory, Trajectory)
    assert len(trajectory) == 6 and trajectory.n_episodes == 2
    assert trajectory.observations.shape == (6, 2, 1)
    assert trajectory.actions.shape == (6, 2, 1)
    assert trajectory.agent_rewards.shape == (6, 2)
    np.testing.assert_array_equal(trajectory.episode_ids, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(trajectory.terminated, [0, 0, 1, 0, 0, 1])
    np.testing.assert_array_equal(trajectory.next_observations[:, 0, 0], [1, 2, 3, 1, 2, 3])
    np.testing.assert_array_equal(trajectory.episode_returns(), [3.0, 3.0])
    assert len(trajectory.episode(1)) == 3
    with pytest.raises(IndexError):
        trajectory.episode(5)
    steps = rollout(env, n_steps=4)
    assert len(steps) == 4 and steps.n_episodes == 2
    assert len(rollout(env, n_steps=10, n_episodes=1)) == 3


def test_rollout_matches_evaluate_and_round_trips(tmp_path):
    env = env_lib.make("LineMsg-v0", max_iter=10)
    policy = env_lib.baseline_policy(env)
    trajectory = rollout(env, policy, n_episodes=2, seed=5)
    result = evaluate(env, policy, n_episodes=2, seed=5)
    np.testing.assert_allclose(trajectory.episode_returns(), result.returns)
    assert trajectory.env_id == "LineMsg-v0"
    path = trajectory.save(tmp_path / "rollout")
    assert path.suffix == ".npz"
    loaded = Trajectory.load(path)
    assert loaded.env_id == "LineMsg-v0"
    for name in ("observations", "actions", "rewards", "agent_rewards", "episode_ids"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(trajectory, name))


def test_rollout_rejects_vector_envs_and_bad_limits():
    envs = CountdownVectorEnv([2, 3])
    with pytest.raises(TypeError):
        rollout(envs)
    with pytest.raises(ValueError):
        rollout(CountdownEnv(), n_steps=0)


def test_array_rewards_are_summed():
    env = env_lib.make("AJLATT-v0", max_episode_steps=3, terminate_on_collision=False)
    result = evaluate(env, n_episodes=1, seed=0)
    assert result.agent_returns.shape == (1, 4)
    assert result.returns[0] == pytest.approx(result.agent_returns.sum())
