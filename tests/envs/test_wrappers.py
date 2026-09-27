"""Tests for env_lib.wrappers: FlattenJointSpaces, TeamReward and the PettingZoo adapter."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

import env_lib
from env_lib.wrappers import (
    PETTINGZOO_AVAILABLE,
    FlattenJointSpaces,
    ParallelEnvAdapter,
    TeamReward,
    to_parallel,
)

# check_env notes that it is given a wrapped environment (intended here) and that
# the observation spaces are unbounded (by design of the environments).
pytestmark = [
    pytest.mark.filterwarnings("ignore:.*different from the unwrapped:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*Box observation space m:UserWarning"),
]


class JointToyEnv(gym.Env):
    """Three agents with per-agent reward and termination arrays (like AJLATT).

    Agent ``i`` terminates at step ``crash[i]`` (``-1``: never); the episode is
    truncated after ``horizon`` steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, crash=(-1, 2, -1), horizon=4, action_space=None, observation_space=None):
        self.crash = np.asarray(crash)
        self.horizon = horizon
        default = spaces.Box(-1.0, 1.0, (3, 2), np.float32)
        self.observation_space = default if observation_space is None else observation_space
        self.action_space = default if action_space is None else action_space
        self.t = 0
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.observation_space.sample() * 0, {"step": 0}

    def step(self, action):
        self.last_action = action
        self.t += 1
        rewards = np.array([1.0, 2.0, 3.0])
        terminated = self.crash == self.t
        info = {"per_agent": np.arange(3) * 10, "shared": "x", "step": self.t}
        obs = np.zeros(self.observation_space.shape, np.float32)
        return obs, rewards, terminated, self.t >= self.horizon, info


# ---------------------------------------------------------------------------
# FlattenJointSpaces
# ---------------------------------------------------------------------------
def test_flatten_box_spaces_and_equivalent_dynamics():
    flat = FlattenJointSpaces(env_lib.make("Consensus-v0"))
    joint = env_lib.make("Consensus-v0")
    assert flat.observation_space.shape == (128,) and flat.observation_space.dtype == np.float32
    assert flat.action_space.shape == (16,)
    obs_flat, _ = flat.reset(seed=0)
    obs_joint, _ = joint.reset(seed=0)
    np.testing.assert_array_equal(obs_flat, obs_joint.reshape(-1))
    action = np.linspace(-1, 1, 16, dtype=np.float32)
    obs_flat, r1, *_ = flat.step(action)
    obs_joint, r2, *_ = joint.step(action.reshape(8, 2))
    np.testing.assert_array_equal(obs_flat, obs_joint.reshape(-1))
    assert r1 == r2
    with pytest.raises(ValueError):
        flat.step(np.zeros(5))


@pytest.mark.parametrize(
    ("env_id", "kwargs"),
    [
        ("Consensus-v0", {}),
        ("KuramotoOscillator-Constant-v0", {}),
        ("LineMsg-v0", {}),
        ("LineMsg-v0", {"action_space_type": "multibinary"}),
        ("WirelessComm-v1", {}),
    ],
)
def test_flattened_envs_pass_gymnasium_checker(env_id, kwargs):
    env = FlattenJointSpaces(env_lib.make(env_id, **kwargs))
    check_env(env, skip_render_check=True)
    assert len(env.observation_space.shape) == 1
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.step(env.action_space.sample())


def test_flatten_discrete_spaces():
    wireless = FlattenJointSpaces(env_lib.make("WirelessComm-v1"))
    assert isinstance(wireless.action_space, spaces.MultiDiscrete)
    assert wireless.action_space.shape == (16,)
    line = FlattenJointSpaces(env_lib.make("LineMsg-v0"))
    assert line.action_space == spaces.Discrete(1024)  # passed through
    toy = FlattenJointSpaces(
        JointToyEnv(
            observation_space=spaces.MultiDiscrete([[3, 4], [5, 6], [7, 8]]),
            action_space=spaces.MultiBinary([3, 2]),
        )
    )
    assert toy.observation_space.shape == (6,)
    np.testing.assert_array_equal(toy.observation_space.high, [2, 3, 4, 5, 6, 7])
    assert toy.action_space == spaces.MultiBinary(6)
    toy.reset()
    toy.step(np.ones(6, dtype=np.int8))
    assert toy.unwrapped.last_action.shape == (3, 2)


def test_flatten_rejects_structured_spaces():
    with pytest.raises(TypeError, match="observation"):
        FlattenJointSpaces(JointToyEnv(observation_space=spaces.Dict({"a": spaces.Discrete(2)})))
    with pytest.raises(TypeError, match="action"):
        FlattenJointSpaces(JointToyEnv(action_space=spaces.Tuple((spaces.Discrete(2),))))


def test_flatten_pistonball_discrete():
    pytest.importorskip("pymunk")
    env = FlattenJointSpaces(env_lib.make("Pistonball-v0", n_pistons=5, continuous=False))
    assert env.observation_space.shape == (35,)
    env.reset(seed=0)
    env.step(env.action_space.sample())


# ---------------------------------------------------------------------------
# TeamReward
# ---------------------------------------------------------------------------
def test_team_reward_reduces_arrays():
    env = TeamReward(JointToyEnv(), reduce="sum")
    env.reset()
    _, reward, terminated, truncated, info = env.step(np.zeros((3, 2)))
    assert reward == 6.0 and terminated is False and truncated is False
    np.testing.assert_array_equal(info["agent_rewards"], [1.0, 2.0, 3.0])
    _, reward, terminated, _, info = env.step(np.zeros((3, 2)))
    assert terminated is True
    np.testing.assert_array_equal(info["agent_terminated"], [False, True, False])
    mean = TeamReward(JointToyEnv(), reduce="mean")
    mean.reset()
    assert mean.step(np.zeros((3, 2)))[1] == 2.0
    with pytest.raises(ValueError):
        TeamReward(JointToyEnv(), reduce="max")


def test_team_reward_scalar_rewards_and_agent_reward_source():
    env = TeamReward(env_lib.make("Consensus-v0"))
    env.reset(seed=0)
    raw = env_lib.make("Consensus-v0")
    raw.reset(seed=0)
    action = np.zeros((8, 2), np.float32)
    assert env.step(action)[1] == raw.step(action)[1]  # scalar team reward is unchanged
    mean = TeamReward(env_lib.make("LineMsg-v0"), reduce="mean", source="agent_rewards")
    mean.reset(seed=0)
    _, reward, _, _, info = mean.step(1023)
    assert reward == pytest.approx(info["agent_rewards"].mean())


def test_team_reward_on_ajlatt_and_in_sync_vector_env():
    env = TeamReward(env_lib.make("AJLATT-v0", max_episode_steps=5))
    check_env(env, skip_render_check=True)
    envs = env_lib.make_vec(
        "AJLATT-v0",
        num_envs=2,
        vectorization_mode="sync",
        wrappers=[TeamReward],
        max_episode_steps=5,
    )
    envs.reset(seed=0)
    _, rewards, terminated, _, infos = envs.step(envs.action_space.sample())
    assert rewards.shape == (2,) and terminated.dtype == bool
    assert infos["agent_rewards"].shape == (2, 4)
    envs.close()


# ---------------------------------------------------------------------------
# ParallelEnvAdapter
# ---------------------------------------------------------------------------
def test_parallel_adapter_spaces_and_step():
    env = to_parallel("Formation-v0")
    assert env.possible_agents == [f"agent_{i}" for i in range(8)]
    assert env.observation_space("agent_3") is env.observation_space("agent_3")
    assert env.observation_space("agent_0").shape == (16,)
    assert env.action_space("agent_0") == spaces.Box(-1.0, 1.0, (2,), np.float32)
    observations, infos = env.reset(seed=0)
    assert set(observations) == set(infos) == set(env.agents)
    np.testing.assert_array_equal(env.state(), env.stack_observations(observations))
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    assert set(rewards) == set(env.possible_agents)
    assert all(isinstance(value, float) for value in rewards.values())
    # Rewards are the per-agent rewards; array infos are sliced per agent.
    assert rewards["agent_3"] == pytest.approx(infos["agent_3"]["agent_rewards"])
    assert infos["agent_2"]["adjacency"].shape == (8,)
    assert infos["agent_2"]["error"] == infos["agent_5"]["error"]
    assert repr(env) == "ParallelEnvAdapter(Formation-v0, 8 agents)"
    env.close()


def test_parallel_adapter_per_agent_termination_and_end_of_episode():
    env = ParallelEnvAdapter(JointToyEnv(crash=(-1, 2, -1), horizon=5))
    env.reset()
    actions = {agent: np.zeros(2, np.float32) for agent in env.agents}
    _, rewards, terminations, truncations, infos = env.step(actions)
    assert rewards == {"agent_0": 1.0, "agent_1": 2.0, "agent_2": 3.0}
    assert not any(terminations.values()) and not any(truncations.values())
    assert infos["agent_1"]["per_agent"] == 10 and infos["agent_1"]["shared"] == "x"
    _, _, terminations, truncations, _ = env.step(actions)
    assert terminations == {"agent_0": False, "agent_1": True, "agent_2": False}
    # The joint episode is over: the other agents are cut off (truncated).
    assert truncations == {"agent_0": True, "agent_1": False, "agent_2": True}
    assert env.agents == []
    with pytest.raises(RuntimeError):
        env.step(actions)
    env.reset()
    with pytest.raises(ValueError, match="missing actions"):
        env.step({"agent_0": np.zeros(2)})


def test_parallel_adapter_action_kinds():
    line = to_parallel("LineMsg-v0", num_agents=4)
    assert line.action_space("agent_0") == spaces.Discrete(2)
    line.reset(seed=0)
    line.step({"agent_0": 1, "agent_1": 0, "agent_2": 1, "agent_3": 1})
    assert line.env.unwrapped.actions[1:5].tolist() == [1, 0, 1, 1]  # bit i = agent i
    assert line.split_actions(0b1101) == {
        "agent_0": 1,
        "agent_1": 0,
        "agent_2": 1,
        "agent_3": 1,
    }
    wireless = to_parallel("WirelessComm-v1")
    assert wireless.action_space("agent_5") == spaces.Discrete(5)
    kuramoto = to_parallel("KuramotoOscillator-Constant-v0")
    assert kuramoto.possible_agents == ["agent_0"]
    assert kuramoto.action_space("agent_0").shape == (6,)
    kuramoto.reset(seed=0)
    _, rewards, *_ = kuramoto.step({"agent_0": np.zeros(6, np.float32)})
    assert set(rewards) == {"agent_0"}
    with pytest.raises(TypeError, match="cannot split"):
        ParallelEnvAdapter(JointToyEnv(action_space=spaces.Box(-1, 1, (4,), np.float32)))
    with pytest.raises(TypeError):
        to_parallel(env_lib.make("LineMsg-v0"), num_agents=3)
    with pytest.raises(TypeError, match="single environment"):
        to_parallel(env_lib.make_vec("LineMsg-v0", num_envs=2, vectorization_mode="sync"))


def test_parallel_adapter_subclasses_pettingzoo_when_available():
    pettingzoo = pytest.importorskip("pettingzoo")
    assert PETTINGZOO_AVAILABLE
    env = to_parallel("LineMsg-v0")
    assert isinstance(env, pettingzoo.utils.env.ParallelEnv)
    assert env.unwrapped is env and env.num_agents == 10 and env.max_num_agents == 10


@pytest.mark.parametrize(
    ("env_id", "kwargs"),
    [
        ("LineMsg-v0", {"max_iter": 20}),
        ("LineMsg-v0", {"action_space_type": "multibinary", "max_iter": 20}),
        ("WirelessComm-v1", {"max_iter": 20}),
        ("AJLATT-v0", {"max_episode_steps": 12}),
        ("Pistonball-v0", {"n_pistons": 5, "max_cycles": 15}),
    ],
)
def test_pettingzoo_parallel_api_test(env_id, kwargs):
    parallel_test = pytest.importorskip("pettingzoo.test")
    if env_id == "Pistonball-v0":
        pytest.importorskip("pymunk")
    parallel_test.parallel_api_test(to_parallel(env_id, **kwargs), num_cycles=100)


class _DropResetOptions(gym.Wrapper):
    """Ignore reset options (parallel_api_test passes a dummy ``{"options": 1}``)."""

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed)


@pytest.mark.parametrize(
    ("env_id", "kwargs"),
    [
        ("Consensus-v0", {"max_steps": 20}),
        ("KuramotoOscillator-Constant-v0", {"max_steps": 20}),
        ("PowerGrid-v0", {}),
        ("Platoon-v0", {}),
    ],
)
def test_pettingzoo_parallel_api_test_strict_option_envs(env_id, kwargs):
    # These environments reject unknown reset options, including the dummy one
    # of parallel_api_test; everything else of the API is checked here.
    parallel_test = pytest.importorskip("pettingzoo.test")
    try:
        env = env_lib.make(env_id, **kwargs)
    except ImportError:
        pytest.skip(f"{env_id} is not available")
    parallel_test.parallel_api_test(ParallelEnvAdapter(_DropResetOptions(env)), num_cycles=30)
