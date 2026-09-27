"""Command line of marl_algorithms: ``marl-train`` (or ``python -m marl_algorithms``).

Examples::

    marl-train list                                  # algorithms
    marl-train presets                               # tuned (algorithm, environment) pairs
    marl-train run mappo PowerGrid-v0                # train with the preset, then evaluate
    marl-train run qmix LineMsg-v0 --steps 50000 --set lr=1e-3
    marl-train run maddpg Consensus-v0 --save runs/maddpg.pt --csv runs/maddpg.csv
    marl-train evaluate runs/maddpg.pt Consensus-v0 --episodes 64
    marl-train compare PowerGrid-v0 --seeds 0 1 2    # every preset algorithm as a baseline
    marl-train compare LineMsg-v0 --algos iql qmix --csv results/linemsg.csv
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import time
from typing import Any

__all__ = ["main"]


class _UsageError(Exception):
    """A command-line mistake, reported as ``marl-train: error: ...`` with exit status 2."""


def _parse_pairs(pairs: list[str] | None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise _UsageError(f"expected KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            values[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            values[key.strip()] = raw
    return values


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {text}")
    return value


def _action_kind(env_id: str, env_kwargs: dict[str, Any]) -> str:
    """Action kind of ``env_id`` built with ``env_kwargs``; checks the id and the arguments."""
    import env_lib
    from marl_algorithms.core.runner import make_vector_env
    from marl_algorithms.core.spec import MultiAgentSpec

    if env_id not in env_lib.list_envs():
        raise _UsageError(f"unknown environment {env_id!r}; see `env-lib list`")
    try:
        envs = make_vector_env(env_id, 1, **env_kwargs)
    except (TypeError, ValueError) as exc:
        raise _UsageError(f"invalid environment arguments for {env_id}: {exc}") from exc
    try:
        return MultiAgentSpec.from_env(envs).action_kind
    finally:
        envs.close()


def _check_output(path: str | None, *, directory: bool = False) -> None:
    """Fail before training if ``path`` cannot be written (a parent is a file)."""
    if not path:
        return
    target = os.path.abspath(path)
    if not directory and os.path.isdir(target):
        raise _UsageError(f"cannot write {path!r}: it is a directory")
    parent = target if directory else os.path.dirname(target)
    while parent and not os.path.exists(parent):
        parent = os.path.dirname(parent)
    if parent and not os.path.isdir(parent):
        raise _UsageError(f"cannot write {path!r}: {parent!r} is not a directory")


def _algorithm_class(name: str) -> Any:
    from marl_algorithms.registry import get_algorithm, list_algorithms

    try:
        return get_algorithm(name)
    except KeyError:
        known = ", ".join(info.name for info in list_algorithms())
        raise _UsageError(f"unknown algorithm {name!r}; choose from {known}") from None


def _check_config(algo_class: Any, config: dict[str, Any]) -> None:
    config_class = algo_class.config_class
    unknown = sorted(set(config) - set(config_class.field_names()))
    if unknown:
        raise _UsageError(
            f"unknown {config_class.__name__} field(s) {', '.join(unknown)}; "
            f"valid fields: {', '.join(config_class.field_names())}"
        )
    try:
        config_class(**config)
    except (TypeError, ValueError) as exc:
        raise _UsageError(f"invalid {config_class.__name__} value: {exc}") from exc


def _cmd_list(_: argparse.Namespace) -> int:
    from marl_algorithms.registry import list_algorithms

    print(f"{'NAME':8s}{'FAMILY':26s}{'ACTIONS':24s}SUMMARY")
    for info in list_algorithms():
        print(f"{info.name:8s}{info.family:26s}{' and '.join(info.action_kinds):24s}{info.summary}")
    print()
    for info in list_algorithms():
        print(f"{info.name:8s}{info.reference}")
    return 0


def _cmd_presets(_: argparse.Namespace) -> int:
    from marl_algorithms.presets import get_preset, list_presets

    print(f"{'ALGORITHM':11s}{'ENVIRONMENT':28s}{'STEPS':>10s}{'COPIES':>8s}")
    for algorithm, env_id in list_presets():
        preset = get_preset(algorithm, env_id) or {}
        print(
            f"{algorithm:11s}{env_id:28s}{preset.get('total_steps', 0):>10,d}{preset.get('num_envs', 16):>8d}"
        )
    return 0


def _evaluation_table(
    algo: Any, env_id: str, env_kwargs: dict[str, Any], episodes: int, seed: int
) -> None:
    import env_lib
    from env_lib.utils.evaluation import evaluate
    from marl_algorithms.core.runner import make_vector_env

    envs = make_vector_env(env_id, min(episodes, 64), **env_kwargs)
    try:
        algo.check_env(envs)
    except ValueError as exc:
        envs.close()
        raise _UsageError(str(exc)) from None
    rows = [("random", evaluate(envs, None, n_episodes=episodes, seed=seed))]
    rows.append((algo.name, algo.evaluate(envs, episodes, seed=seed)))
    try:
        baseline = env_lib.baseline_policy(envs)
    except (TypeError, NotImplementedError):
        baseline = None
    if baseline is not None:
        rows.append(("baseline", evaluate(envs, baseline, n_episodes=episodes, seed=seed)))
    envs.close()
    print(f"\nEvaluation on {env_id}, {episodes} episodes (seed {seed}):")
    print(f"  {'policy':10s}{'mean return':>14s}{'95% CI +-':>12s}{'length':>9s}")
    for label, result in rows:
        print(
            f"  {label:10s}{result.mean_return:>14.4g}{result.ci95:>12.3g}{result.lengths.mean():>9.1f}"
        )


def _error(message: str) -> int:
    """Print a command-line error to stderr; returns the exit status 2."""
    print(f"marl-train: error: {message}", file=sys.stderr)
    return 2


def _cmd_run(args: argparse.Namespace) -> int:

    from marl_algorithms.presets import get_preset
    from marl_algorithms.registry import train

    algo_class = _algorithm_class(args.algorithm)
    preset = None if args.no_preset else get_preset(args.algorithm, args.env_id)
    preset = preset or {}
    env_kwargs = {**preset.get("env_kwargs", {}), **_parse_pairs(args.env_kwarg)}
    config = {**preset.get("config", {}), **_parse_pairs(args.set)}
    action_kind = _action_kind(args.env_id, env_kwargs)
    if action_kind not in algo_class.action_kinds:
        raise _UsageError(
            f"{args.algorithm} does not support the {action_kind} actions of {args.env_id}"
        )
    _check_config(algo_class, config)
    _check_output(args.save)
    _check_output(args.csv)
    total_steps = args.steps or int(preset.get("total_steps", 100_000))
    num_envs = args.num_envs or int(preset.get("num_envs", 16))
    source = "preset" if preset else "defaults"
    print(
        f"Training {args.algorithm} on {args.env_id} ({source}): {total_steps:,} env steps, "
        f"{num_envs} copies, seed {args.seed}"
    )
    start = time.perf_counter()
    reports = max(1, args.reports)
    next_report = [total_steps / reports]

    def progress(algo: Any, log: Any) -> None:
        if algo.env_steps >= next_report[0]:
            next_report[0] += total_steps / reports
            print(
                f"  {algo.env_steps:>10,d} steps  mean return (last 20) {log.mean_return(20):>12.4g}  "
                f"{time.perf_counter() - start:6.1f} s"
            )

    algo, log = train(
        args.algorithm,
        args.env_id,
        total_steps,
        num_envs=num_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
        callback=progress,
        threads=args.threads or None,
        **config,
    )
    print(log.summary())
    if args.save:
        print(f"saved {algo.save(args.save)}")
    if args.csv:
        print(f"saved {log.to_csv(args.csv)}")
    if args.eval_episodes > 0:
        _evaluation_table(algo, args.env_id, env_kwargs, args.eval_episodes, args.seed + 1)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:

    from marl_algorithms.baselines import compare

    env_kwargs = _parse_pairs(args.env_kwarg)
    config = _parse_pairs(args.set)
    _action_kind(args.env_id, env_kwargs)  # the id and the arguments
    _check_output(args.csv)
    _check_output(args.save_dir, directory=True)
    try:  # compare() checks everything else before training
        report = compare(
            args.env_id,
            args.algos,
            seeds=args.seeds,
            n_episodes=args.episodes,
            eval_seed=args.eval_seed,
            total_steps=args.steps,
            num_envs=args.num_envs,
            env_kwargs=env_kwargs,
            threads=args.threads or None,
            verbose=True,
            **config,
        )
    except (KeyError, ValueError) as exc:
        raise _UsageError(str(exc.args[0]) if exc.args else str(exc)) from exc
    print()
    print(report.to_markdown() if args.markdown else report)
    if args.csv:
        print(f"\nsaved {report.to_csv(args.csv)}")
    if args.save_dir:
        for (name, seed), algo in report.algorithms.items():
            print(f"saved {algo.save(os.path.join(args.save_dir, f'{name}_seed{seed}.pt'))}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from marl_algorithms.core.base import Algorithm

    env_kwargs = _parse_pairs(args.env_kwarg)
    _action_kind(args.env_id, env_kwargs)
    if not os.path.isfile(args.checkpoint):
        raise _UsageError(f"no checkpoint at {args.checkpoint!r}")
    try:
        algo = Algorithm.load(args.checkpoint)
    except (ValueError, KeyError) as exc:
        raise _UsageError(str(exc.args[0]) if exc.args else str(exc)) from None
    trained_on = algo.metadata.get("env_id")
    trained_kwargs = algo.metadata.get("env_kwargs", {})
    if trained_on is not None and (trained_on, trained_kwargs) != (args.env_id, env_kwargs):
        extra = f" with {trained_kwargs}" if trained_kwargs else ""
        print(
            f"marl-train: note: the checkpoint was trained on {trained_on}{extra}",
            file=sys.stderr,
        )
    _evaluation_table(algo, args.env_id, env_kwargs, args.episodes, args.seed)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``marl-train``."""
    parser = argparse.ArgumentParser(
        prog="marl-train",
        description="Train classical multi-agent RL algorithms on the env_lib environments.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    sub.add_parser("list", help="list the algorithms").set_defaults(func=_cmd_list)
    sub.add_parser("presets", help="list the tuned presets").set_defaults(func=_cmd_presets)

    run = sub.add_parser("run", help="train an algorithm, then evaluate it")
    run.add_argument("algorithm", help="ippo, mappo, maddpg, matd3, iql, vdn or qmix")
    run.add_argument("env_id", help="env_lib environment id")
    run.add_argument(
        "--steps", type=_positive_int, default=None, help="environment steps (default: preset)"
    )
    run.add_argument(
        "--num-envs", type=_positive_int, default=None, help="batched copies (default: preset)"
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-preset", action="store_true", help="ignore the tuned preset")
    run.add_argument("--set", nargs="+", metavar="KEY=VALUE", help="configuration overrides")
    run.add_argument("--env-kwarg", nargs="+", metavar="KEY=VALUE", help="environment arguments")
    run.add_argument(
        "--eval-episodes", type=_non_negative_int, default=64, help="evaluation episodes (0 skips)"
    )
    run.add_argument(
        "--reports", type=_positive_int, default=10, help="progress lines during training"
    )
    run.add_argument("--save", metavar="PATH", help="save the trained algorithm")
    run.add_argument("--csv", metavar="PATH", help="save the training episodes as CSV")
    run.add_argument(
        "--threads", type=_non_negative_int, default=1, help="PyTorch threads (0 keeps the default)"
    )
    run.set_defaults(func=_cmd_run)

    compare = sub.add_parser(
        "compare",
        help="train baseline algorithms and compare them with random actions and the "
        "classical controller",
    )
    compare.add_argument("env_id", help="env_lib environment id")
    compare.add_argument(
        "--algos", nargs="+", metavar="ALGO", help="algorithms (default: all with a preset)"
    )
    compare.add_argument("--seeds", nargs="+", type=int, default=[0], help="training seeds [0]")
    compare.add_argument(
        "--steps", type=_positive_int, default=None, help="environment steps per run"
    )
    compare.add_argument(
        "--num-envs", type=_positive_int, default=None, help="batched copies per run"
    )
    compare.add_argument(
        "--episodes", type=_positive_int, default=64, help="evaluation episodes [64]"
    )
    compare.add_argument("--eval-seed", type=int, default=1, help="evaluation seed [1]")
    compare.add_argument("--set", nargs="+", metavar="KEY=VALUE", help="configuration overrides")
    compare.add_argument(
        "--env-kwarg", nargs="+", metavar="KEY=VALUE", help="environment arguments"
    )
    compare.add_argument("--markdown", action="store_true", help="print a Markdown table")
    compare.add_argument("--csv", metavar="PATH", help="save the report as CSV")
    compare.add_argument("--save-dir", metavar="DIR", help="save every trained algorithm")
    compare.add_argument("--threads", type=_non_negative_int, default=1, help="PyTorch threads [1]")
    compare.set_defaults(func=_cmd_compare)

    evaluate = sub.add_parser("evaluate", help="evaluate a saved algorithm")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("env_id")
    evaluate.add_argument("--episodes", type=_positive_int, default=64)
    evaluate.add_argument("--seed", type=int, default=1)
    evaluate.add_argument("--env-kwarg", nargs="+", metavar="KEY=VALUE")
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``marl-train``."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except _UsageError as exc:
        return _error(str(exc))
    except BrokenPipeError:  # output piped into a command that exited (e.g. `| head`)
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print(
            'marl-train: error: PyTorch is not installed; install it with pip install "my-tool-box[torch]"',
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
