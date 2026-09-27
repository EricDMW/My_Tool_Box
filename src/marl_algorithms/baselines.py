"""The integrated algorithms and classical controllers as baselines for new methods.

:func:`compare` answers the question every new method faces: how does it do
against the established ones? It trains the algorithms of this package that
have a tuned preset for an environment (or the ones you name), with one or
several seeds, and evaluates them together with random actions, the
environment's classical controller (``env_lib.baseline_policy``) and any
policies you pass -- all on the same seeded evaluation episodes, so the
numbers are directly comparable::

    from marl_algorithms import compare

    report = compare("PowerGrid-v0", seeds=(0, 1, 2), policies={"mine": my_policy})
    print(report)                          # one row per method, best first on request
    report.to_csv("results/power_grid.csv")
    mappo = report.algorithms[("mappo", 0)]  # the trained baselines, for reuse

The command line offers the same: ``marl-train compare PowerGrid-v0 --seeds 0 1 2``.
With the default single seed 0 and evaluation seed 1, the numbers are those of
the published results table (``benchmarks/benchmark_marl.py``).
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from marl_algorithms.core.base import Algorithm, TrainingLog
from marl_algorithms.core.runner import make_vector_env
from marl_algorithms.presets import get_preset, list_presets, train_preset
from marl_algorithms.registry import get_algorithm, train

__all__ = ["Comparison", "ComparisonRow", "compare", "per_copy"]

Policy = Callable[[Any], Any]


@dataclass(frozen=True)
class ComparisonRow:
    """Result of one method in a :class:`Comparison`.

    Attributes
    ----------
    name:
        ``"random"``, ``"baseline"`` (the classical controller), an algorithm
        name or the name of a policy you passed.
    kind:
        ``"reference"`` (random actions, classical controller), ``"algorithm"``
        (trained by :func:`compare`) or ``"policy"`` (yours).
    returns:
        Mean evaluation return of every training seed (one value for
        references and your policies).
    seeds:
        The training seeds (empty for references and your policies).
    env_steps:
        Training steps per seed (algorithms only).
    train_seconds:
        Wall-clock training time per seed (algorithms only).
    """

    name: str
    kind: str
    returns: tuple[float, ...]
    seeds: tuple[int, ...] = ()
    env_steps: tuple[int, ...] = ()
    train_seconds: tuple[float, ...] = ()

    @property
    def mean(self) -> float:
        """Mean return over the seeds."""
        return float(np.mean(self.returns))

    @property
    def std(self) -> float:
        """Standard deviation of the return over the seeds (NaN for one value)."""
        return float(np.std(self.returns, ddof=1)) if len(self.returns) > 1 else math.nan


@dataclass
class Comparison:
    """Report of :func:`compare`: one row per method, in the order evaluated.

    Attributes
    ----------
    env_id, env_kwargs:
        The environment, as trained and evaluated.
    n_episodes, eval_seed:
        The evaluation episodes shared by every row.
    rows:
        :class:`ComparisonRow` of every method: references first, then the
        algorithms, then your policies.
    algorithms, logs:
        The trained algorithms and their training logs, keyed by
        ``(name, seed)``, for saving, rendering or further evaluation.
    """

    env_id: str
    env_kwargs: dict[str, Any]
    n_episodes: int
    eval_seed: int | None
    rows: list[ComparisonRow] = field(default_factory=list)
    algorithms: dict[tuple[str, int], Algorithm] = field(default_factory=dict)
    logs: dict[tuple[str, int], TrainingLog] = field(default_factory=dict)

    def __getitem__(self, name: str) -> ComparisonRow:
        for row in self.rows:
            if row.name == name:
                return row
        raise KeyError(f"no row {name!r}; rows: {[row.name for row in self.rows]}")

    def ranking(self, kinds: Sequence[str] = ("algorithm", "policy")) -> list[ComparisonRow]:
        """Rows of the given kinds, highest mean return first."""
        return sorted((r for r in self.rows if r.kind in kinds), key=lambda r: -r.mean)

    def records(self) -> list[dict[str, Any]]:
        """One flat dictionary per row (for pandas, JSON or logging)."""
        return [
            {
                "env_id": self.env_id,
                "name": row.name,
                "kind": row.kind,
                "mean_return": row.mean,
                "std_return": row.std,
                "n_seeds": len(row.seeds),
                "returns": list(row.returns),
                "seeds": list(row.seeds),
                "env_steps": int(np.mean(row.env_steps)) if row.env_steps else None,
                "train_seconds": float(np.mean(row.train_seconds)) if row.train_seconds else None,
                "n_episodes": self.n_episodes,
                "eval_seed": self.eval_seed,
            }
            for row in self.rows
        ]

    def to_csv(self, path: str | Path) -> Path:
        """Write :meth:`records` as CSV (``returns`` and ``seeds`` joined by ``;``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self.records()
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["name"])
            writer.writeheader()
            for record in records:
                record = dict(record)
                record["returns"] = ";".join(f"{value:.6g}" for value in record["returns"])
                record["seeds"] = ";".join(str(seed) for seed in record["seeds"])
                writer.writerow(record)
        return path

    def to_markdown(self) -> str:
        """The report as a Markdown table."""
        lines = [
            "| Method | Kind | Mean return | Std over seeds | Seeds | Env steps | Train time [s] |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for cells in self._cells():
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def __str__(self) -> str:
        header = ("method", "kind", "mean return", "std", "seeds", "env steps", "train [s]")
        rows = [header, *self._cells()]
        widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
        title = (
            f"{self.env_id}: mean team return over {self.n_episodes} evaluation episodes "
            f"(seed {self.eval_seed}); higher is better"
        )
        out = [title, ""]
        for k, row in enumerate(rows):
            cells = [row[0].ljust(widths[0]), row[1].ljust(widths[1])]
            cells += [cell.rjust(width) for cell, width in zip(row[2:], widths[2:])]
            out.append("  ".join(cells))
            if k == 0:
                out.append("  ".join("-" * width for width in widths))
        return "\n".join(out)

    def _cells(self) -> list[tuple[str, ...]]:
        cells = []
        for row in self.rows:
            steps = f"{int(np.mean(row.env_steps)):,}" if row.env_steps else "-"
            seconds = f"{np.mean(row.train_seconds):.1f}" if row.train_seconds else "-"
            std = _fmt(row.std) if not math.isnan(row.std) else "-"
            n_seeds = str(len(row.seeds)) if row.seeds else "-"
            cells.append((row.name, row.kind, _fmt(row.mean), std, n_seeds, steps, seconds))
        return cells


def compare(
    env_id: str,
    algorithms: Sequence[str] | None = None,
    *,
    seeds: Sequence[int] = (0,),
    policies: Mapping[str, Policy | Algorithm] | None = None,
    n_episodes: int = 64,
    eval_seed: int | None = 1,
    total_steps: int | None = None,
    num_envs: int | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
    include_random: bool = True,
    include_baseline: bool = True,
    device: str = "cpu",
    verbose: bool = False,
    **config: Any,
) -> Comparison:
    """Train baseline algorithms on ``env_id`` and compare them with your policies.

    Parameters
    ----------
    env_id:
        ``env_lib`` environment id.
    algorithms:
        Algorithm names to train (see ``marl-train list``). By default every
        algorithm with a preset for ``env_id`` (``marl-train presets``); an
        empty sequence trains none, to compare only your policies with the
        references.
    seeds:
        Training seeds; every algorithm is trained once per seed and the
        report gives the mean and standard deviation over seeds.
    policies:
        Your methods, by name: callables mapping the batched observations of
        a vector environment, ``(num_envs, n_agents, obs_dim)``, to batched
        actions (wrap a single-environment policy with :func:`per_copy`), or
        trained :class:`~marl_algorithms.core.base.Algorithm` instances.
    n_episodes, eval_seed:
        Evaluation episodes, shared by every method: a vector environment of
        ``min(n_episodes, 64)`` copies reset with ``eval_seed``.
    total_steps, num_envs:
        Training budget and batched copies for every algorithm, replacing the
        presets' values. Algorithms without a preset for ``env_id`` train
        with their default configuration and need ``total_steps``.
    env_kwargs:
        Environment arguments for training and evaluation (merged into the
        presets' arguments). The presets' hyperparameters are kept, so they
        may need more steps on a larger variant.
    include_random, include_baseline:
        Add rows for uniformly random actions and for the classical
        controller (skipped for environments without one).
    device:
        Torch device for training.
    verbose:
        Print a line per trained algorithm and seed.
    **config:
        Configuration overrides applied to every algorithm (fields shared by
        their configuration classes, such as ``gamma``).

    Returns
    -------
    Comparison

    Raises
    ------
    KeyError
        For an unknown algorithm.
    ValueError
        For an algorithm without a preset when ``total_steps`` is not given, a
        preset whose environment arguments differ from ``env_kwargs``, a
        policy name that repeats another row's name, empty ``seeds`` or an
        invalid ``n_episodes``.
    """
    from env_lib.baselines import baseline_policy
    from env_lib.utils.evaluation import evaluate

    if isinstance(n_episodes, bool) or int(n_episodes) != n_episodes or n_episodes < 1:
        raise ValueError(f"n_episodes must be a positive integer, got {n_episodes!r}")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("seeds must not be empty")
    if algorithms is None:
        names = [name for name, preset_env in list_presets() if preset_env == env_id]
    else:
        names = [str(name).lower() for name in algorithms]
    env_kwargs = dict(env_kwargs or {})
    for name in names:
        get_algorithm(name)  # KeyError for unknown names
        preset = get_preset(name, env_id)
        if preset is None and total_steps is None:
            raise ValueError(
                f"{name} has no preset for {env_id}; pass total_steps to train it with its "
                "default configuration"
            )
        if preset is not None and {**preset.get("env_kwargs", {}), **env_kwargs} != env_kwargs:
            raise ValueError(
                f"the {name} preset uses the environment arguments {preset['env_kwargs']}; "
                "pass them in env_kwargs so that every method sees the same environment"
            )
    policies = dict(policies or {})
    taken = {"random", "baseline", *names}
    for label in policies:
        if label in taken:
            raise ValueError(f"policy name {label!r} repeats the name of another row")
    report = Comparison(env_id, env_kwargs, int(n_episodes), eval_seed)
    envs = make_vector_env(env_id, min(int(n_episodes), 64), **env_kwargs)
    try:
        if include_random:
            result = evaluate(envs, None, n_episodes=n_episodes, seed=eval_seed)
            report.rows.append(ComparisonRow("random", "reference", (result.mean_return,)))
        if include_baseline:
            try:
                controller = baseline_policy(envs)
            except (TypeError, NotImplementedError):
                controller = None
            if controller is not None:
                result = evaluate(envs, controller, n_episodes=n_episodes, seed=eval_seed)
                report.rows.append(ComparisonRow("baseline", "reference", (result.mean_return,)))

        # Your policies are evaluated before the (long) training runs, so that
        # an error in one of them shows at once; their rows come last.
        policy_rows = []
        for label, policy in policies.items():
            if isinstance(policy, Algorithm):
                result = policy.evaluate(envs, int(n_episodes), seed=eval_seed)
            else:
                result = evaluate(envs, policy, n_episodes=n_episodes, seed=eval_seed)
            policy_rows.append(ComparisonRow(str(label), "policy", (result.mean_return,)))

        for name in names:
            returns, steps, seconds = [], [], []
            for seed in seeds:
                start = time.perf_counter()
                algo, log = _train(
                    name, env_id, seed, total_steps, num_envs, env_kwargs, device, config
                )
                seconds.append(time.perf_counter() - start)
                result = algo.evaluate(envs, int(n_episodes), seed=eval_seed)
                returns.append(result.mean_return)
                steps.append(int(algo.env_steps))
                report.algorithms[(name, seed)] = algo
                report.logs[(name, seed)] = log
                if verbose:
                    print(
                        f"  {name:8s} seed {seed}: return {_fmt(result.mean_return):>10s} "
                        f"after {algo.env_steps:,} env steps in {seconds[-1]:.1f} s",
                        flush=True,
                    )
            report.rows.append(
                ComparisonRow(
                    name, "algorithm", tuple(returns), seeds, tuple(steps), tuple(seconds)
                )
            )
        report.rows.extend(policy_rows)
    finally:
        envs.close()
    return report


def per_copy(policy: Policy) -> Policy:
    """Apply a single-environment policy to every copy of a vector environment.

    ``per_copy(policy)(obs)`` stacks ``policy(obs[i])`` over the copies ``i``,
    so a policy written for one environment can be passed to :func:`compare`
    or ``env_lib.evaluate`` with a vector environment. Batched policies are
    faster; use this for policies that cannot be batched.
    """

    def batched(observations: Any) -> np.ndarray:
        return np.stack([np.asarray(policy(obs)) for obs in observations])

    return batched


def _train(
    name: str,
    env_id: str,
    seed: int,
    total_steps: int | None,
    num_envs: int | None,
    env_kwargs: dict[str, Any],
    device: str,
    config: dict[str, Any],
) -> tuple[Algorithm, TrainingLog]:
    preset = get_preset(name, env_id)
    if preset is not None:
        return train_preset(
            name,
            env_id,
            seed=seed,
            total_steps=total_steps,
            num_envs=num_envs,
            env_kwargs=env_kwargs,
            device=device,
            **config,
        )
    return train(
        name,
        env_id,
        int(total_steps),  # checked by compare()
        num_envs=num_envs if num_envs is not None else 16,
        seed=seed,
        env_kwargs=env_kwargs,
        device=device,
        **config,
    )


def _fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}" if abs(value) < 1 else f"{value:.2f}"
