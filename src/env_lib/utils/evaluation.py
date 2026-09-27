"""Policy evaluation and trajectory recording.

Two functions cover the everyday measurement tasks of an experiment:

* :func:`evaluate` runs a policy for a number of episodes and summarises the
  returns (mean, standard deviation, 95 % confidence interval), episode
  lengths, termination and success rates and per-agent returns. It accepts a
  single environment and a vector environment (a native batched one created by
  :func:`env_lib.make_vec` or a Gymnasium ``SyncVectorEnv``/``AsyncVectorEnv``),
  in which case the episodes run in parallel.
* :func:`rollout` records the transitions of a single environment as stacked
  NumPy arrays (:class:`Trajectory`), which can be saved to and loaded from an
  ``.npz`` file.

A *policy* is any callable mapping an observation to an action, for example a
baseline controller from :func:`env_lib.baseline_policy` or a wrapped neural
network. ``policy=None`` samples uniformly random actions.

Examples
--------
>>> import env_lib
>>> env = env_lib.make("Consensus-v0")
>>> result = env_lib.evaluate(env, env_lib.baseline_policy(env), n_episodes=5, seed=0)
>>> print(result)                                       # doctest: +SKIP
>>> envs = env_lib.make_vec("Consensus-v0", num_envs=16)
>>> result = env_lib.evaluate(envs, env_lib.baseline_policy(envs), n_episodes=64, seed=0)
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
from gymnasium.vector import VectorEnv

__all__ = ["EvaluationResult", "Trajectory", "evaluate", "rollout"]

Policy = Callable[[Any], Any]
PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_numpy(value: Any) -> Any:
    """Convert a torch tensor (duck-typed) to a NumPy array; leave anything else unchanged."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return value


def _team_reward(reward: Any) -> float:
    """Scalar team reward: per-agent reward arrays are summed."""
    reward = _to_numpy(reward)
    if np.ndim(reward) == 0:
        return float(reward)
    return float(np.sum(reward))


def _any(flag: Any) -> bool:
    return bool(np.any(_to_numpy(flag)))


def _agent_rewards(reward: Any, info: Mapping[str, Any]) -> np.ndarray | None:
    """Per-agent rewards of one step: ``info["agent_rewards"]`` or an array reward."""
    value = info.get("agent_rewards") if isinstance(info, Mapping) else None
    if value is None and np.ndim(_to_numpy(reward)) > 0:
        value = reward
    if value is None:
        return None
    return np.asarray(_to_numpy(value), dtype=np.float64).reshape(-1)


def _env_id(env: Any) -> str | None:
    spec = getattr(env, "spec", None)
    if spec is None:
        spec = getattr(getattr(env, "unwrapped", None), "spec", None)
    return getattr(spec, "id", None)


def _policy_name(policy: Policy | None) -> str:
    if policy is None:
        return "random"
    name = getattr(policy, "name", None)
    if isinstance(name, str) and name:
        return name
    return getattr(policy, "__name__", type(policy).__name__)


def _random_policy(space: Any, seed: int | None) -> Policy:
    """Uniformly random actions from a private, seeded copy of ``space``."""
    space = copy.deepcopy(space)
    space.seed(seed)
    return lambda _observation: space.sample()


def _autoreset_mode(env: VectorEnv) -> str:
    """``"next_step"``, ``"same_step"`` or ``"disabled"`` for a vector environment."""
    mode = env.metadata.get("autoreset_mode", getattr(env, "autoreset_mode", "next_step"))
    raw = str(getattr(mode, "value", mode)).replace("_", "").lower()
    return {"nextstep": "next_step", "samestep": "same_step", "disabled": "disabled"}.get(
        raw, "next_step"
    )


def _t_quantile(dof: int) -> float:
    """0.975 quantile of Student's t distribution with ``dof`` degrees of freedom."""
    from scipy.special import stdtrit  # lighter import than scipy.stats

    return float(stdtrit(dof, 0.975))


def _fmt(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if value != 0.0 and (abs(value) >= 1e6 or abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.3f}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class EvaluationResult:
    """Summary of a policy evaluation.

    Attributes
    ----------
    returns:
        Team return of every episode, shape ``(n_episodes,)``. Per-agent reward
        arrays (AJLATT) are summed.
    lengths:
        Episode lengths in steps, shape ``(n_episodes,)``.
    terminated:
        Whether each episode ended by termination (task success or failure,
        depending on the environment) rather than truncation.
    agent_returns:
        Per-agent returns, shape ``(n_episodes, n_agents)``, from
        ``info["agent_rewards"]``; ``None`` if the environment does not report
        them.
    successes:
        ``info["success"]`` at the end of each episode, or ``None`` if the
        environment does not report success.
    env_id, policy:
        Labels used by :meth:`__str__`.
    n_episodes, mean_return, std_return, ci95, mean_length, termination_rate, success_rate:
        Derived statistics. ``std_return`` is the sample standard deviation
        (``ddof=1``; 0 for a single episode) and ``ci95`` the half-width of the
        95 % Student-t confidence interval of the mean return (``nan`` for a
        single episode). ``success_rate`` is ``None`` when ``successes`` is.
    """

    returns: np.ndarray
    lengths: np.ndarray
    terminated: np.ndarray
    agent_returns: np.ndarray | None = None
    successes: np.ndarray | None = None
    env_id: str | None = None
    policy: str | None = None
    n_episodes: int = field(init=False)
    mean_return: float = field(init=False)
    std_return: float = field(init=False)
    ci95: float = field(init=False)
    mean_length: float = field(init=False)
    termination_rate: float = field(init=False)
    success_rate: float | None = field(init=False)

    def __post_init__(self) -> None:
        returns = np.array(self.returns, dtype=np.float64).reshape(-1)
        n = returns.size
        if n == 0:
            raise ValueError("an EvaluationResult needs at least one episode")
        arrays = {
            "returns": returns,
            "lengths": np.array(self.lengths, dtype=np.int64).reshape(n),
            "terminated": np.array(self.terminated, dtype=bool).reshape(n),
        }
        if self.agent_returns is not None:
            arrays["agent_returns"] = np.array(self.agent_returns, dtype=np.float64).reshape(n, -1)
        if self.successes is not None:
            arrays["successes"] = np.array(self.successes, dtype=bool).reshape(n)
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        std = float(returns.std(ddof=1)) if n > 1 else 0.0
        derived = {
            "n_episodes": n,
            "mean_return": float(returns.mean()),
            "std_return": std,
            "ci95": _t_quantile(n - 1) * std / math.sqrt(n) if n > 1 else math.nan,
            "mean_length": float(arrays["lengths"].mean()),
            "termination_rate": float(arrays["terminated"].mean()),
            "success_rate": None if self.successes is None else float(self.successes.mean()),
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)

    def summary(self) -> dict[str, Any]:
        """The scalar statistics as a plain dictionary (handy for logging)."""
        out: dict[str, Any] = {
            "env_id": self.env_id,
            "policy": self.policy,
            "n_episodes": self.n_episodes,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "ci95": self.ci95,
            "mean_length": self.mean_length,
            "termination_rate": self.termination_rate,
            "success_rate": self.success_rate,
        }
        if self.agent_returns is not None:
            out["mean_agent_returns"] = self.agent_returns.mean(axis=0).tolist()
        return out

    def __str__(self) -> str:
        head = "EvaluationResult"
        labels = [
            label for label in (self.env_id, self.policy and f"policy {self.policy}") if label
        ]
        labels.append(f"{self.n_episodes} episode{'s' if self.n_episodes != 1 else ''}")
        lines = [f"{head}: {', '.join(labels)}"]
        lines.append(
            f"  return        {_fmt(self.mean_return)} +- {_fmt(self.std_return)} (std), "
            f"95% CI +- {_fmt(self.ci95)}"
        )
        lines.append(
            f"  length        {self.mean_length:.1f} steps "
            f"(min {int(self.lengths.min())}, max {int(self.lengths.max())})"
        )
        lines.append(f"  terminated    {100.0 * self.termination_rate:.1f}% of episodes")
        if self.success_rate is not None:
            lines.append(f"  success       {100.0 * self.success_rate:.1f}% of episodes")
        if self.agent_returns is not None:
            per_agent = self.agent_returns.mean(axis=0)
            lines.append(
                f"  agent return  mean {_fmt(float(per_agent.mean()))}, "
                f"min {_fmt(float(per_agent.min()))}, max {_fmt(float(per_agent.max()))} "
                f"({per_agent.size} agent{'s' if per_agent.size != 1 else ''})"
            )
        return "\n".join(lines)


class _Episodes:
    """Accumulates finished episodes."""

    def __init__(self) -> None:
        self.returns: list[float] = []
        self.lengths: list[int] = []
        self.terminated: list[bool] = []
        self.agent_returns: list[np.ndarray | None] = []
        self.successes: list[bool | None] = []

    def add(
        self,
        ret: float,
        length: int,
        terminated: bool,
        agent_return: np.ndarray | None,
        success: Any,
    ) -> None:
        self.returns.append(float(ret))
        self.lengths.append(int(length))
        self.terminated.append(bool(terminated))
        self.agent_returns.append(None if agent_return is None else np.array(agent_return))
        self.successes.append(None if success is None else bool(success))

    def result(self, env_id: str | None, policy: str) -> EvaluationResult:
        agent = self.agent_returns
        agent_returns = None
        if agent and all(a is not None for a in agent) and len({a.shape for a in agent}) == 1:
            agent_returns = np.stack(agent)
        successes = None
        if self.successes and all(s is not None for s in self.successes):
            successes = np.array(self.successes, dtype=bool)
        return EvaluationResult(
            returns=np.array(self.returns),
            lengths=np.array(self.lengths),
            terminated=np.array(self.terminated),
            agent_returns=agent_returns,
            successes=successes,
            env_id=env_id,
            policy=policy,
        )


def evaluate(
    env: Any,
    policy: Policy | None = None,
    *,
    n_episodes: int = 10,
    seed: int | None = None,
    max_steps: int | None = None,
    deterministic_seeds: bool = True,
) -> EvaluationResult:
    """Run ``policy`` for ``n_episodes`` episodes and summarise the results.

    Parameters
    ----------
    env:
        A Gymnasium environment or vector environment. Vector environments run
        their copies in parallel: copy ``i`` contributes
        ``(n_episodes + i) // num_envs`` episodes (the first ones it finishes),
        so short episodes are not over-represented. All three autoreset modes
        are supported; under ``"next_step"`` the reset step that follows the
        end of an episode is not counted.
    policy:
        Callable mapping an observation (batched for vector environments) to
        an action. ``None`` samples uniformly random actions from a private,
        seeded copy of the action space.
    n_episodes:
        Number of episodes to evaluate (``>= 1``).
    seed:
        Base seed. Single environment: episode ``k`` is reset with
        ``seed + k`` (``deterministic_seeds=True``) or only the first reset is
        seeded. Vector environment: the batch is reset once with
        ``reset(seed=seed)`` and later episodes follow from automatic resets.
        The random policy is seeded with ``seed`` as well.
    max_steps:
        Optional cap on the episode length; capped episodes count as
        truncated. Vector copies that reach the cap are reset through
        ``reset(options={"reset_mask": ...})``.
    deterministic_seeds:
        See ``seed``. Ignored for vector environments.

    Returns
    -------
    EvaluationResult

    Raises
    ------
    ValueError
        If ``n_episodes < 1`` or ``max_steps < 1``.

    Examples
    --------
    >>> import env_lib
    >>> env = env_lib.make("LineMsg-v0")
    >>> random_result = env_lib.evaluate(env, n_episodes=5, seed=0)
    >>> baseline = env_lib.evaluate(env, env_lib.baseline_policy(env), n_episodes=5, seed=0)
    >>> baseline.mean_return > random_result.mean_return
    True
    """
    if isinstance(n_episodes, bool) or int(n_episodes) != n_episodes or n_episodes < 1:
        raise ValueError(f"n_episodes must be a positive integer, got {n_episodes!r}")
    if max_steps is not None and (int(max_steps) != max_steps or max_steps < 1):
        raise ValueError(f"max_steps must be a positive integer or None, got {max_steps!r}")
    act = policy if policy is not None else _random_policy(env.action_space, seed)
    if isinstance(env, VectorEnv):
        episodes = _evaluate_vector(env, act, int(n_episodes), seed, max_steps)
    else:
        episodes = _evaluate_single(env, act, int(n_episodes), seed, max_steps, deterministic_seeds)
    return episodes.result(_env_id(env), _policy_name(policy))


def _episode_seed(seed: int | None, episode: int, deterministic: bool) -> int | None:
    if seed is None:
        return None
    if deterministic:
        return int(seed) + episode
    return int(seed) if episode == 0 else None


def _evaluate_single(
    env: Any,
    act: Policy,
    n_episodes: int,
    seed: int | None,
    max_steps: int | None,
    deterministic_seeds: bool,
) -> _Episodes:
    episodes = _Episodes()
    for episode in range(n_episodes):
        observation, info = env.reset(seed=_episode_seed(seed, episode, deterministic_seeds))
        total, length, agent_total = 0.0, 0, None
        missing_agent_rewards = False
        while True:
            observation, reward, terminated, truncated, info = env.step(act(observation))
            length += 1
            total += _team_reward(reward)
            agent = _agent_rewards(reward, info)
            if agent is None:
                missing_agent_rewards = True
            elif agent_total is None:
                agent_total = agent.copy()
            elif agent_total.shape == agent.shape:
                agent_total += agent
            else:
                missing_agent_rewards = True
            ended = _any(terminated)
            if ended or _any(truncated) or (max_steps is not None and length >= max_steps):
                break
        episodes.add(
            total,
            length,
            ended,
            None if missing_agent_rewards else agent_total,
            info.get("success") if isinstance(info, Mapping) else None,
        )
    return episodes


def _batched_value(infos: Mapping[str, Any], key: str, num_envs: int) -> tuple[Any, np.ndarray]:
    """Batched info value and its validity mask (all false when missing)."""
    value = infos.get(key)
    if value is None:
        return None, np.zeros(num_envs, dtype=bool)
    mask = infos.get(f"_{key}")
    mask = np.ones(num_envs, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return value, mask


def _final_info_value(infos: Mapping[str, Any], key: str, index: int) -> Any:
    """``key`` of the final info of copy ``index`` (``"same_step"`` autoreset)."""
    final = infos.get("final_info")
    if final is None:
        return None
    if isinstance(final, Mapping):  # Gymnasium Sync/Async convention: a batched dict
        value, mask = _batched_value(final, key, index + 1)
        if value is None or not mask[index]:
            return None
        return value[index]
    entry = final[index]  # object array of per-copy dictionaries
    return entry.get(key) if isinstance(entry, Mapping) else None


def _step_agent_rewards(
    infos: Mapping[str, Any], rewards: np.ndarray, mode: str, done: np.ndarray, num_envs: int
) -> tuple[np.ndarray | None, np.ndarray]:
    """Per-agent rewards of one vector step ``(num_envs, n_agents)`` and their validity.

    Under ``"same_step"`` autoreset the info of a copy that was reset within
    the step describes the new episode; its last reward is read from
    ``infos["final_info"]`` instead.
    """
    value, mask = _batched_value(infos, "agent_rewards", num_envs)
    valid = mask.copy()
    values: np.ndarray | None = None
    if value is not None:
        values = np.array(_to_numpy(value), dtype=np.float64).reshape(num_envs, -1)
    elif rewards.shape[1] > 1:  # per-agent reward arrays
        values, valid = rewards.copy(), np.ones(num_envs, dtype=bool)
    if mode == "same_step":
        for index in np.flatnonzero(done):
            final = _final_info_value(infos, "agent_rewards", int(index))
            if final is None:
                continue
            final = np.asarray(_to_numpy(final), dtype=np.float64).reshape(-1)
            if values is None:
                values = np.zeros((num_envs, final.size))
            if final.size == values.shape[1]:
                values[index] = final
                valid[index] = True
    return values, valid


def _evaluate_vector(
    env: VectorEnv, act: Policy, n_episodes: int, seed: int | None, max_steps: int | None
) -> _Episodes:
    num_envs = int(env.num_envs)
    mode = _autoreset_mode(env)
    targets = np.array([(n_episodes + i) // num_envs for i in range(num_envs)])
    counts = np.zeros(num_envs, dtype=np.int64)
    episodes = _Episodes()

    observations, _ = env.reset(seed=seed)
    returns = np.zeros(num_envs)
    lengths = np.zeros(num_envs, dtype=np.int64)
    agent_returns: np.ndarray | None = None
    agent_missing = np.zeros(num_envs, dtype=bool)
    pending = np.zeros(num_envs, dtype=bool)  # next_step mode: copy resets on this step

    while np.any(counts < targets):
        observations, rewards, terminations, truncations, infos = env.step(act(observations))
        rewards = np.asarray(_to_numpy(rewards), dtype=np.float64).reshape(num_envs, -1)
        terminated = np.asarray(terminations, dtype=bool).reshape(num_envs, -1).any(axis=1)
        truncated = np.asarray(truncations, dtype=bool).reshape(num_envs, -1).any(axis=1)
        active = ~pending
        returns[active] += rewards[active].sum(axis=1)
        lengths[active] += 1
        done = active & (terminated | truncated)

        step_agent, valid = _step_agent_rewards(infos, rewards, mode, done, num_envs)
        if step_agent is None:
            agent_missing |= active
        else:
            if agent_returns is None:
                agent_returns = np.zeros((num_envs, step_agent.shape[1]))
            if step_agent.shape[1] == agent_returns.shape[1]:
                use = active & valid
                agent_returns[use] += step_agent[use]
                agent_missing |= active & ~valid
            else:
                agent_missing |= active

        capped = active & ~done
        if max_steps is not None:
            capped &= lengths >= max_steps
        else:
            capped[:] = False
        finished = done | capped
        if finished.any():
            success_value, success_mask = _batched_value(infos, "success", num_envs)
            for index in np.flatnonzero(finished):
                i = int(index)
                if counts[i] < targets[i]:
                    success = None
                    if mode == "same_step" and done[i]:
                        success = _final_info_value(infos, "success", i)
                    elif success_value is not None and success_mask[i]:
                        success = success_value[i]
                    agent = None
                    if agent_returns is not None and not agent_missing[i]:
                        agent = agent_returns[i].copy()
                    episodes.add(returns[i], lengths[i], bool(terminated[i]), agent, success)
                    counts[i] += 1
                returns[i] = 0.0
                lengths[i] = 0
                agent_missing[i] = False
                if agent_returns is not None:
                    agent_returns[i] = 0.0

        pending = done if mode == "next_step" else np.zeros(num_envs, dtype=bool)
        to_reset = finished if mode == "disabled" else capped
        if to_reset.any() and np.any(counts < targets):
            observations, _ = env.reset(options={"reset_mask": to_reset.copy()})
    return episodes


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------
_TRAJECTORY_ARRAYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminated",
    "truncated",
    "episode_ids",
)


@dataclass
class Trajectory:
    """Transitions of one or more episodes, stacked along the first axis.

    Attributes
    ----------
    observations:
        Observation before each step, shape ``(T, *observation_shape)``.
    actions:
        Action of each step, shape ``(T, *action_shape)``.
    rewards:
        Team reward of each step (per-agent reward arrays summed), shape ``(T,)``.
    next_observations:
        Observation after each step, shape ``(T, *observation_shape)``.
    terminated, truncated:
        Episode-end flags of each step (per-agent flags reduced with ``any``).
    episode_ids:
        Index of the episode each step belongs to, shape ``(T,)``.
    agent_rewards:
        Per-agent rewards ``info["agent_rewards"]``, shape ``(T, n_agents)``,
        or ``None`` when the environment does not report them.
    env_id:
        Registered id of the environment, if known.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_ids: np.ndarray
    agent_rewards: np.ndarray | None = None
    env_id: str | None = None

    def __len__(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def n_episodes(self) -> int:
        """Number of (possibly incomplete) episodes in the trajectory."""
        return int(np.unique(self.episode_ids).size)

    def episode_returns(self) -> np.ndarray:
        """Team return of every episode, in episode order."""
        ids = np.unique(self.episode_ids)
        return np.array([self.rewards[self.episode_ids == i].sum() for i in ids])

    def episode(self, index: int) -> Trajectory:
        """The transitions of episode ``index``."""
        select = self.episode_ids == index
        if not select.any():
            raise IndexError(f"trajectory has no episode {index}")
        return Trajectory(
            **{name: getattr(self, name)[select] for name in _TRAJECTORY_ARRAYS},
            agent_rewards=None if self.agent_rewards is None else self.agent_rewards[select],
            env_id=self.env_id,
        )

    def save(self, path: PathLike) -> Path:
        """Write the trajectory to a compressed ``.npz`` file and return its path."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(path.suffix + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self, name) for name in _TRAJECTORY_ARRAYS}
        if self.agent_rewards is not None:
            arrays["agent_rewards"] = self.agent_rewards
        if self.env_id is not None:
            arrays["env_id"] = np.array(self.env_id)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: PathLike) -> Trajectory:
        """Read a trajectory written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            arrays = {name: data[name] for name in _TRAJECTORY_ARRAYS}
            agent_rewards = data["agent_rewards"] if "agent_rewards" in data.files else None
            env_id = str(data["env_id"]) if "env_id" in data.files else None
        return cls(**arrays, agent_rewards=agent_rewards, env_id=env_id)


def rollout(
    env: Any,
    policy: Policy | None = None,
    *,
    n_steps: int | None = None,
    n_episodes: int | None = None,
    seed: int | None = None,
) -> Trajectory:
    """Record transitions of a single environment.

    Parameters
    ----------
    env:
        A Gymnasium environment (not a vector environment).
    policy:
        Callable mapping an observation to an action; ``None`` samples random
        actions (seeded with ``seed``).
    n_steps:
        Stop after this many steps (the last episode may be incomplete).
    n_episodes:
        Stop after this many episodes. With neither limit one episode is
        recorded; with both, whichever is reached first ends the rollout.
    seed:
        Episode ``k`` is reset with ``seed + k``.

    Returns
    -------
    Trajectory

    Raises
    ------
    TypeError
        For a vector environment.
    ValueError
        For non-positive limits.

    Examples
    --------
    >>> import env_lib
    >>> env = env_lib.make("Consensus-v0")
    >>> trajectory = env_lib.rollout(env, env_lib.baseline_policy(env), n_episodes=2, seed=0)
    >>> trajectory.observations.shape[1:]
    (8, 16)
    >>> path = trajectory.save("renders/consensus_rollout.npz")     # doctest: +SKIP
    """
    if isinstance(env, VectorEnv):
        raise TypeError(
            "rollout() records a single environment; use evaluate() for vector environments "
            "or record one copy with env_lib.make(...)"
        )
    for name, value in (("n_steps", n_steps), ("n_episodes", n_episodes)):
        if value is not None and (isinstance(value, bool) or int(value) != value or value < 1):
            raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
    if n_steps is None and n_episodes is None:
        n_episodes = 1
    act = policy if policy is not None else _random_policy(env.action_space, seed)

    columns: dict[str, list[Any]] = {name: [] for name in _TRAJECTORY_ARRAYS}
    agent_rows: list[np.ndarray | None] = []
    episode, steps = 0, 0
    observation, _ = env.reset(seed=_episode_seed(seed, 0, True))
    while True:
        action = act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        columns["observations"].append(np.array(_to_numpy(observation)))
        columns["actions"].append(np.array(_to_numpy(action)))
        columns["rewards"].append(_team_reward(reward))
        columns["next_observations"].append(np.array(_to_numpy(next_observation)))
        columns["terminated"].append(_any(terminated))
        columns["truncated"].append(_any(truncated))
        columns["episode_ids"].append(episode)
        agent_rows.append(_agent_rewards(reward, info))
        steps += 1
        observation = next_observation
        if n_steps is not None and steps >= n_steps:
            break
        if _any(terminated) or _any(truncated):
            episode += 1
            if n_episodes is not None and episode >= n_episodes:
                break
            observation, _ = env.reset(seed=_episode_seed(seed, episode, True))

    agent_rewards = None
    if all(row is not None for row in agent_rows) and len({row.shape for row in agent_rows}) == 1:
        agent_rewards = np.stack(agent_rows)
    return Trajectory(
        observations=np.stack(columns["observations"]),
        actions=np.stack(columns["actions"]),
        rewards=np.array(columns["rewards"], dtype=np.float64),
        next_observations=np.stack(columns["next_observations"]),
        terminated=np.array(columns["terminated"], dtype=bool),
        truncated=np.array(columns["truncated"], dtype=bool),
        episode_ids=np.array(columns["episode_ids"], dtype=np.int64),
        agent_rewards=agent_rewards,
        env_id=_env_id(env),
    )
