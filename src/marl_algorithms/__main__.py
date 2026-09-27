"""Command line of marl_algorithms: ``marl-train`` (or ``python -m marl_algorithms``).

Examples::

    marl-train list                                  # algorithms
    marl-train presets                               # tuned (algorithm, environment) pairs
    marl-train run mappo PowerGrid-v0                # train with the preset, then evaluate
    marl-train run qmix LineMsg-v0 --steps 50000 --set lr=1e-3
    marl-train run maddpg Consensus-v0 --save runs/maddpg.pt --csv runs/maddpg.csv
    marl-train evaluate runs/maddpg.pt Consensus-v0 --episodes 64
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import time
from typing import Any

__all__ = ["main"]


def _parse_pairs(pairs: list[str] | None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"expected KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            values[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            values[key.strip()] = raw
    return values


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
    import torch

    import env_lib
    from marl_algorithms.presets import get_preset
    from marl_algorithms.registry import get_algorithm, list_algorithms, train

    try:
        algo_class = get_algorithm(args.algorithm)
    except KeyError:
        names = ", ".join(info.name for info in list_algorithms())
        return _error(f"unknown algorithm {args.algorithm!r}; choose from {names}")
    if args.env_id not in env_lib.list_envs():
        return _error(f"unknown environment {args.env_id!r}; see `env-lib list`")
    preset = None if args.no_preset else get_preset(args.algorithm, args.env_id)
    preset = preset or {}
    env_kwargs = {**preset.get("env_kwargs", {}), **_parse_pairs(args.env_kwarg)}
    config = {**preset.get("config", {}), **_parse_pairs(args.set)}
    unknown = sorted(set(config) - set(algo_class.config_class.field_names()))
    if unknown:
        return _error(
            f"unknown {algo_class.config_class.__name__} field(s) {', '.join(unknown)}; "
            f"valid fields: {', '.join(algo_class.config_class.field_names())}"
        )
    if args.threads:
        torch.set_num_threads(args.threads)
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


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import env_lib
    from marl_algorithms.core.base import Algorithm

    if args.env_id not in env_lib.list_envs():
        return _error(f"unknown environment {args.env_id!r}; see `env-lib list`")
    if not os.path.isfile(args.checkpoint):
        return _error(f"no checkpoint at {args.checkpoint!r}")
    algo = Algorithm.load(args.checkpoint)
    _evaluation_table(algo, args.env_id, _parse_pairs(args.env_kwarg), args.episodes, args.seed)
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
    run.add_argument("--steps", type=int, default=None, help="environment steps (default: preset)")
    run.add_argument("--num-envs", type=int, default=None, help="batched copies (default: preset)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-preset", action="store_true", help="ignore the tuned preset")
    run.add_argument("--set", nargs="+", metavar="KEY=VALUE", help="configuration overrides")
    run.add_argument("--env-kwarg", nargs="+", metavar="KEY=VALUE", help="environment arguments")
    run.add_argument("--eval-episodes", type=int, default=64, help="evaluation episodes (0 skips)")
    run.add_argument("--reports", type=int, default=10, help="progress lines during training")
    run.add_argument("--save", metavar="PATH", help="save the trained algorithm")
    run.add_argument("--csv", metavar="PATH", help="save the training episodes as CSV")
    run.add_argument("--threads", type=int, default=1, help="PyTorch threads (0 keeps the default)")
    run.set_defaults(func=_cmd_run)

    evaluate = sub.add_parser("evaluate", help="evaluate a saved algorithm")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("env_id")
    evaluate.add_argument("--episodes", type=int, default=64)
    evaluate.add_argument("--seed", type=int, default=1)
    evaluate.add_argument("--env-kwarg", nargs="+", metavar="KEY=VALUE")
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``marl-train``."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except BrokenPipeError:  # output piped into a command that exited (e.g. `| head`)
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(main())
