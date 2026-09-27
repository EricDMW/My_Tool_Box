"""Command-line interface of ``env_lib`` (``env-lib`` or ``python -m env_lib``).

Examples
--------
.. code-block:: text

    env-lib list --continuous
    env-lib describe Formation-v0
    env-lib run Consensus-v0 --policy baseline --seed 0 --gif renders/consensus.gif
    env-lib run Formation-v0 --kwarg formation_shape=wedge topology=proximity
    env-lib evaluate WirelessComm-v1 --episodes 20
    env-lib bench Consensus-v0 --num-envs 256 --steps 200
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import math
import os
import sys
import time
import warnings
from collections.abc import Sequence
from typing import Any, Callable

import gymnasium as gym
import numpy as np

__all__ = ["main"]

_MAX_AGENT_ROWS = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _CliError(Exception):
    """Error reported to the user without a traceback."""


def _parse_kwargs(groups: Sequence[Sequence[str]] | None) -> dict[str, Any]:
    """``["n_agents=16", "task=formation"]`` -> ``{"n_agents": 16, "task": "formation"}``."""
    kwargs: dict[str, Any] = {}
    for item in (entry for group in groups or () for entry in group):
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise _CliError(f"--kwarg expects key=value, got {item!r}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        kwargs[key.strip()] = value
    return kwargs


def _spec(env_id: str) -> Any:
    from env_lib.registration import get_spec, list_envs

    try:
        return get_spec(env_id)
    except KeyError:
        close = difflib.get_close_matches(env_id, list_envs(), n=3)
        hint = f"; did you mean {', '.join(close)}?" if close else "; run 'env-lib list'"
        raise _CliError(f"unknown environment {env_id!r}{hint}") from None


def _make(env_id: str, kwargs: dict[str, Any], **extra: Any) -> gym.Env:
    import env_lib

    _spec(env_id)
    try:
        return env_lib.make(env_id, **kwargs, **extra)
    except ImportError as exc:
        raise _CliError(f"cannot create {env_id}: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise _CliError(f"invalid arguments for {env_id}: {exc}") from exc


def _policy(env: Any, name: str, seed: int | None) -> tuple[Callable[[Any], Any] | None, str]:
    """``(policy, label)``; ``None`` stands for random actions in ``evaluate``."""
    if name == "random":
        return None, "random: uniform actions"
    from env_lib.baselines import baseline_policy

    try:
        policy = baseline_policy(env)
    except (TypeError, NotImplementedError) as exc:
        raise _CliError(f"no baseline controller available: {exc}") from exc
    return policy, f"baseline: {policy.description}"


def _random_actions(space: gym.Space, seed: int | None) -> Callable[[Any], Any]:
    space = copy.deepcopy(space)
    space.seed(seed)
    return lambda _observation: space.sample()


def _number(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    if value != 0.0 and (abs(value) >= 1e6 or abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.4f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], right: Sequence[int] = ()) -> str:
    """Aligned plain-text table; columns listed in ``right`` are right-aligned."""
    cells = [list(header), *[list(row) for row in rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    lines = []
    for row in cells:
        parts = [
            cell.rjust(width) if i in right else cell.ljust(width)
            for i, (cell, width) in enumerate(zip(row, widths))
        ]
        lines.append("  ".join(parts).rstrip())
    return "\n".join(lines)


class _EpisodeRecorder(gym.Wrapper):
    """Accumulate the statistics of the current episode."""

    def reset(self, **kwargs: Any) -> Any:
        self.steps = 0
        self.team_return = 0.0
        self.agent_return: np.ndarray | None = None
        self.terminated = False
        self.truncated = False
        self.last_info: dict[str, Any] = {}
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Any:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.team_return += float(np.sum(reward))
        agent = info.get("agent_rewards")
        if agent is None and np.ndim(reward) > 0:
            agent = reward
        if agent is not None:
            agent = np.asarray(agent, dtype=np.float64).reshape(-1)
            if self.agent_return is None:
                self.agent_return = agent.copy()
            elif self.agent_return.shape == agent.shape:
                self.agent_return += agent
        self.terminated = bool(np.any(terminated))
        self.truncated = bool(np.any(truncated))
        self.last_info = dict(info)
        return observation, reward, terminated, truncated, info


def _scalar_info(info: dict[str, Any], limit: int = 8) -> str:
    parts = []
    for key, value in info.items():
        if key.startswith("_") or key == "agent_rewards":
            continue
        if isinstance(value, (bool, np.bool_)):
            parts.append(f"{key}={bool(value)}")
        elif isinstance(value, (int, np.integer)):
            parts.append(f"{key}={int(value)}")
        elif isinstance(value, (float, np.floating)):
            parts.append(f"{key}={float(value):.4g}")
        if len(parts) >= limit:
            break
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
def _cmd_list(args: argparse.Namespace) -> int:
    from env_lib.catalog import catalog

    action_type = "continuous" if args.continuous else "discrete" if args.discrete else None
    entries = catalog(
        family=args.family,
        action_type=action_type,
        native_vector=True if args.native else None,
        available_only=args.available,
        inspect=not args.no_inspect,
    )
    print(entries.to_markdown() if args.markdown else entries)
    if not args.markdown:
        print(f"\n{len(entries)} environment(s). Details: env-lib describe <ID>")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    from env_lib.catalog import describe

    _spec(args.env_id)
    print(describe(args.env_id))
    return 0


def _cmd_baselines(args: argparse.Namespace) -> int:
    from env_lib.baselines import list_baselines

    rows = [(family, text) for family, text in list_baselines().items()]
    print(_table(("FAMILY", "BASELINE CONTROLLER"), rows))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    kwargs = _parse_kwargs(args.kwarg)
    if args.gif:
        from env_lib.utils.rendering import set_theme

        set_theme(args.theme)
    env = _make(args.env_id, kwargs, render_mode="rgb_array" if args.gif else None)
    recorder = _EpisodeRecorder(env)
    try:
        policy, label = _policy(env, args.policy, args.seed)
        act = policy if policy is not None else _random_actions(env.action_space, args.seed)
        started = time.perf_counter()
        if args.gif:
            from env_lib.utils.recording import record_episode

            frames = record_episode(
                recorder, act, args.gif, max_steps=args.steps, seed=args.seed, fps=args.fps
            )
        else:
            frames = None
            observation, _ = recorder.reset(seed=args.seed)
            while args.steps is None or recorder.steps < args.steps:
                observation, _, terminated, truncated, _ = recorder.step(act(observation))
                if np.any(terminated) or np.any(truncated):
                    break
        elapsed = time.perf_counter() - started
    finally:
        env.close()

    ending = (
        "terminated"
        if recorder.terminated
        else "truncated"
        if recorder.truncated
        else "stopped (--steps)"
    )
    kw_text = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    print(f"{args.env_id}{f' ({kw_text})' if kw_text else ''}")
    print(f"  policy        {label}")
    print(f"  seed          {args.seed}")
    print(f"  episode       {recorder.steps} steps, {ending} ({elapsed:.2f} s)")
    print(f"  team return   {_number(recorder.team_return)}")
    agent = recorder.agent_return
    if agent is not None and agent.size > 1:
        print(
            f"  agent return  mean {_number(float(agent.mean()))}, min {_number(float(agent.min()))} "
            f"(agent_{int(agent.argmin())}), max {_number(float(agent.max()))} "
            f"(agent_{int(agent.argmax())})"
        )
        if agent.size <= _MAX_AGENT_ROWS:
            rows = [(f"agent_{i}", _number(float(value))) for i, value in enumerate(agent)]
            table = _table(("agent", "return"), rows, right=(1,))
            print("\n".join("    " + line for line in table.splitlines()))
    final = _scalar_info(recorder.last_info)
    if final:
        print(f"  final info    {final}")
    if frames is not None:
        print(f"  gif           {args.gif} ({len(frames)} frames)")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import env_lib
    from env_lib.utils.evaluation import evaluate

    kwargs = _parse_kwargs(args.kwarg)
    spec = _spec(args.env_id)
    if args.num_envs > 1:
        extra: dict[str, Any] = {}
        if spec.vector_entry_point is None and spec.disable_env_checker:
            from env_lib.wrappers import TeamReward

            extra["wrappers"] = [TeamReward]
        try:
            env = env_lib.make_vec(args.env_id, args.num_envs, **extra, **kwargs)
        except (ImportError, TypeError, ValueError) as exc:
            raise _CliError(
                f"cannot create {args.num_envs} copies of {args.env_id}: {exc}"
            ) from exc
    else:
        env = _make(args.env_id, kwargs)
    names = ["random", "baseline"] if args.policy == "both" else [args.policy]
    try:
        results = []
        for name in names:
            policy, _ = _policy(env, name, args.seed)
            started = time.perf_counter()
            result = evaluate(env, policy, n_episodes=args.episodes, seed=args.seed)
            results.append((name, result, time.perf_counter() - started))
    finally:
        env.close()
    mode = f"{args.num_envs} parallel copies" if args.num_envs > 1 else "single environment"
    print(f"{args.env_id}: {args.episodes} episodes per policy, seed {args.seed}, {mode}")
    rows = []
    for name, result, seconds in results:
        success = "-" if result.success_rate is None else f"{100 * result.success_rate:.0f}%"
        rows.append(
            (
                name,
                _number(result.mean_return),
                _number(result.std_return),
                _number(result.ci95),
                f"{result.mean_length:.1f}",
                f"{100 * result.termination_rate:.0f}%",
                success,
                f"{seconds:.2f}",
            )
        )
    header = ("policy", "mean return", "std", "95% CI +-", "length", "terminated", "success", "s")
    print(_table(header, rows, right=(1, 2, 3, 4, 5, 6, 7)))
    return 0


def _time_single(env: gym.Env, steps: int, seed: int | None) -> tuple[float, bool]:
    """Seconds for ``steps`` steps with a fixed random action; also whether rewards are arrays."""
    env.reset(seed=seed)
    env.action_space.seed(seed)
    action = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)
    array_reward = np.ndim(reward) > 0
    env.reset(seed=seed)
    started = time.perf_counter()
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(action)
        if np.any(terminated) or np.any(truncated):
            env.reset()
    return time.perf_counter() - started, array_reward


def _time_vector(envs: Any, steps: int, seed: int | None) -> float:
    envs.reset(seed=seed)
    envs.action_space.seed(seed)
    actions = envs.action_space.sample()
    envs.step(actions)
    started = time.perf_counter()
    for _ in range(steps):
        envs.step(actions)
    return time.perf_counter() - started


def _cmd_bench(args: argparse.Namespace) -> int:
    import env_lib
    from env_lib.wrappers import TeamReward

    kwargs = _parse_kwargs(args.kwarg)
    spec = _spec(args.env_id)
    rows: list[tuple[str, ...]] = []
    notes: list[str] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = _make(args.env_id, kwargs)
        try:
            seconds, array_reward = _time_single(env, args.steps, args.seed)
        finally:
            env.close()
    single_rate = args.steps / seconds
    rows.append(("single", "1", str(args.steps), f"{seconds:.3f}", f"{single_rate:,.0f}", "1.0x"))

    modes = [("native", "vector_entry_point"), ("sync", "sync")]
    if args.async_:
        modes.append(("async", "async"))
    for label, mode in modes:
        if mode == "vector_entry_point" and spec.vector_entry_point is None:
            notes.append("native: no native vector implementation registered")
            continue
        extra: dict[str, Any] = {}
        if mode != "vector_entry_point" and array_reward:
            extra["wrappers"] = [TeamReward]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                envs = env_lib.make_vec(
                    args.env_id, args.num_envs, vectorization_mode=mode, **extra, **kwargs
                )
        except Exception as exc:  # report and continue with the other modes
            notes.append(f"{label}: unavailable ({type(exc).__name__}: {exc})")
            continue
        try:
            seconds = _time_vector(envs, args.steps, args.seed)
        finally:
            envs.close()
        rate = args.num_envs * args.steps / seconds
        rows.append(
            (
                label,
                str(args.num_envs),
                str(args.steps),
                f"{seconds:.3f}",
                f"{rate:,.0f}",
                f"{rate / single_rate:.1f}x",
            )
        )
    print(
        f"{args.env_id}: {args.steps} steps per mode with a fixed random action "
        f"(env steps/s = copies x steps / seconds)"
    )
    header = ("mode", "copies", "steps", "seconds", "env steps/s", "speed-up")
    print(_table(header, rows, right=(1, 2, 3, 4, 5)))
    for note in notes:
        print(f"note: {note}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="env-lib",
        description="Discover, run, evaluate and benchmark the env_lib environments.",
        epilog="Run 'env-lib <command> -h' for the options of a command.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True
    kwarg_help = "constructor argument(s) key=value (Python literals), e.g. n_agents=16"

    p = sub.add_parser("list", help="list the registered environments")
    kind = p.add_mutually_exclusive_group()
    kind.add_argument("--continuous", action="store_true", help="continuous actions only")
    kind.add_argument("--discrete", action="store_true", help="discrete actions only")
    p.add_argument(
        "--family", nargs="+", default=None, help="only these families (e.g. consensus platoon)"
    )
    p.add_argument(
        "--native", action="store_true", help="only environments with a native vector env"
    )
    p.add_argument("--available", action="store_true", help="hide unavailable environments")
    p.add_argument("--markdown", action="store_true", help="print a Markdown table")
    p.add_argument(
        "--no-inspect",
        action="store_true",
        help="do not instantiate the environments (faster; no shapes)",
    )
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("describe", help="describe one environment")
    p.add_argument("env_id")
    p.set_defaults(func=_cmd_describe)

    p = sub.add_parser("baselines", help="list the baseline controllers")
    p.set_defaults(func=_cmd_baselines)

    p = sub.add_parser("run", help="roll out one episode and print a summary")
    p.add_argument("env_id")
    p.add_argument("--steps", type=_positive, default=None, help="maximum number of steps")
    p.add_argument("--policy", choices=("baseline", "random"), default="baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", default=None, help="record the episode to this GIF (or .mp4) file")
    p.add_argument("--fps", type=float, default=None, help="playback rate of the recording")
    p.add_argument("--theme", choices=("dark", "light"), default="dark", help="rendering theme")
    p.add_argument("--kwarg", nargs="+", action="append", metavar="KEY=VALUE", help=kwarg_help)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("evaluate", help="compare random actions and the baseline")
    p.add_argument("env_id")
    p.add_argument("--episodes", type=_positive, default=10)
    p.add_argument("--policy", choices=("both", "baseline", "random"), default="both")
    p.add_argument("--num-envs", type=_positive, default=1, help="evaluate copies in parallel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kwarg", nargs="+", action="append", metavar="KEY=VALUE", help=kwarg_help)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("bench", help="measure simulation throughput")
    p.add_argument("env_id")
    p.add_argument("--num-envs", type=_positive, default=32)
    p.add_argument("--steps", type=_positive, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--async", dest="async_", action="store_true", help="also time AsyncVectorEnv")
    p.add_argument("--kwarg", nargs="+", action="append", metavar="KEY=VALUE", help=kwarg_help)
    p.set_defaults(func=_cmd_bench)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``env-lib`` command line.

    Parameters
    ----------
    argv:
        Arguments without the program name (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit status: 0 on success, 2 for usage errors.
    """
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except _CliError as exc:
        print(f"env-lib: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        return 130
    except BrokenPipeError:  # pragma: no cover - output piped into e.g. `head`
        # Silence the error Python would print when flushing stdout at exit.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 1


if __name__ == "__main__":
    sys.exit(main())
