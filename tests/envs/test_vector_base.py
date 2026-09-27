"""Tests for the BatchedVectorEnv base class, using a toy batched environment."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from env_lib.errors import ResetNeededError
from env_lib.utils.vector import BatchedVectorEnv


class CounterVectorEnv(BatchedVectorEnv):
    # Reset info reports how often each copy was reset.
    """Each copy counts up by its action and terminates at ``limit``."""

    def __init__(self, num_envs, limit=3, **kwargs):
        super().__init__(
            num_envs,
            spaces.Box(0.0, np.inf, (1,), np.float32),
            spaces.Box(0.0, 2.0, (1,), np.float32),
            **kwargs,
        )
        self.limit = limit
        self.count = np.zeros(num_envs)
        self.resets = np.zeros(num_envs, dtype=int)

    def _reset_envs(self, mask, options):
        start = 0.0 if options is None else float(options.get("start", 0.0))
        self.count[mask] = start
        self.resets[mask] += 1

    def _step_envs(self, actions, active):
        self.count += actions[:, 0]
        terminated = self.count >= self.limit
        return (
            self.count.copy(),
            terminated,
            np.zeros(self.num_envs, bool),
            {"count": self.count.copy()},
        )

    def _observe(self):
        return self.count[:, None].astype(np.float32)

    def _reset_infos(self, mask):
        return {"resets": self.resets.copy()}


def test_spaces_and_reset():
    env = CounterVectorEnv(4)
    assert env.observation_space.shape == (4, 1)
    assert env.action_space.shape == (4, 1)
    obs, info = env.reset(seed=0, options={"start": 1.0})
    assert obs.shape == (4, 1) and np.all(obs == 1.0)
    assert info["resets"].tolist() == [1, 1, 1, 1] and info["_resets"].all()


def test_step_before_reset_raises():
    env = CounterVectorEnv(2)
    with pytest.raises(ResetNeededError):
        env.step(np.ones((2, 1)))


def test_next_step_autoreset():
    env = CounterVectorEnv(2, limit=2)
    env.reset(seed=0)
    actions = np.array([[2.0], [1.0]], dtype=np.float32)
    obs, rew, term, trunc, info = env.step(actions)
    assert term.tolist() == [True, False]
    assert info["count"].tolist() == [2.0, 1.0] and info["_count"].all()
    obs, rew, term, trunc, info = env.step(actions)
    # Copy 0 is reset (action ignored, zero reward); copy 1 terminates now.
    assert obs[:, 0].tolist() == [0.0, 2.0]
    assert rew.tolist() == [0.0, 2.0]
    assert term.tolist() == [False, True]
    assert env.resets.tolist() == [2, 1]
    # Like SyncVectorEnv: the reset copy reports its reset info, not the step info.
    assert info["_count"].tolist() == [False, True]
    assert info["_resets"].tolist() == [True, False]
    assert info["resets"][0] == 2


def test_same_step_autoreset():
    env = CounterVectorEnv(2, limit=2, autoreset_mode="same_step")
    env.reset(seed=0)
    obs, rew, term, trunc, info = env.step(np.array([[2.0], [1.0]], dtype=np.float32))
    assert term.tolist() == [True, False]
    assert obs[:, 0].tolist() == [0.0, 1.0]
    assert info["_final_obs"].tolist() == [True, False]
    assert info["final_obs"][0].tolist() == [2.0]
    # The main info rows of the reset copy describe its new episode.
    assert info["_count"].tolist() == [False, True]
    assert info["resets"][0] == 2 and info["_resets"].tolist() == [True, False]
    assert info["final_info"]["count"][0] == 2.0
    assert info["final_info"]["_count"].tolist() == [True, False]


def test_disabled_autoreset_and_partial_reset():
    env = CounterVectorEnv(3, limit=1, autoreset_mode="disabled")
    env.reset(seed=0)
    env.step(np.ones((3, 1), dtype=np.float32))
    obs, _ = env.reset(options={"reset_mask": np.array([True, False, True])})
    assert obs[:, 0].tolist() == [0.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        env.reset(options={"reset_mask": np.array([True])})


def test_action_validation():
    env = CounterVectorEnv(2)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.ones((3, 1)))
    with pytest.raises(ValueError):
        env.step(np.array([[np.nan], [1.0]]))
    env.step(np.ones(2))  # reshaped to (2, 1)


def test_autoreset_mode_spellings():
    assert CounterVectorEnv(1, autoreset_mode="SameStep").autoreset_mode == "same_step"
    try:
        from gymnasium.vector import AutoresetMode
    except ImportError:  # Gymnasium 1.0
        return
    env = CounterVectorEnv(1, autoreset_mode=AutoresetMode.DISABLED)
    assert env.autoreset_mode == "disabled"
    assert env.metadata["autoreset_mode"] is AutoresetMode.DISABLED


def test_invalid_arguments():
    with pytest.raises(ValueError):
        CounterVectorEnv(0)
    with pytest.raises(ValueError):
        CounterVectorEnv(2, autoreset_mode="sometimes")
