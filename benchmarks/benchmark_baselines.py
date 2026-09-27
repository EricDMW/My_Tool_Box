"""Random actions against the classical controller on every registered environment.

For every id of ``env_lib.list_envs()`` (default configuration) the script
creates ``env_lib.make_vec(id, EPISODES)`` (native batch where available,
``SyncVectorEnv`` otherwise), resets it with seed 0 and evaluates uniformly
random actions and ``env_lib.baseline_policy`` on the first episode of every
copy, so both policies meet the same initial conditions. AJLATT is also run
with ``terminate_on_collision=False``: its default episodes end on the first
collision, which rewards random robots for colliding early.

Run::

    python benchmarks/benchmark_baselines.py                  # all ids, 64 episodes
    python benchmarks/benchmark_baselines.py --markdown       # table for the README
    python benchmarks/benchmark_baselines.py --only PowerGrid-v0 Platoon-v0 --episodes 16
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import env_lib
from env_lib.utils.evaluation import evaluate

EXTRA_CASES = [("AJLATT-v0", {"terminate_on_collision": False})]


def run_case(env_id: str, kwargs: dict, episodes: int, seed: int) -> dict:
    """Evaluate random actions and the baseline on ``episodes`` first episodes."""
    envs = env_lib.make_vec(env_id, episodes, **kwargs)
    start = time.perf_counter()
    random_result = evaluate(envs, None, n_episodes=episodes, seed=seed)
    try:
        baseline = env_lib.baseline_policy(envs)
    except (TypeError, NotImplementedError):
        baseline = None
    baseline_result = (
        evaluate(envs, baseline, n_episodes=episodes, seed=seed) if baseline is not None else None
    )
    envs.close()
    return {
        "env_id": env_id,
        "kwargs": kwargs,
        "random": random_result,
        "baseline": baseline_result,
        "seconds": time.perf_counter() - start,
    }


def fmt(value: float) -> str:
    if abs(round(value)) >= 1000:
        return f"{value:,.0f}"
    if abs(round(value, 1)) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def label(record: dict) -> str:
    extra = ", ".join(f"{k}={v}" for k, v in record["kwargs"].items())
    return f"{record['env_id']} ({extra})" if extra else record["env_id"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", metavar="ID", help="subset of environment ids")
    parser.add_argument("--episodes", type=int, default=64, help="episodes (= copies) [64]")
    parser.add_argument("--seed", type=int, default=0, help="reset seed [0]")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args()

    cases = [(env_id, {}) for env_id in env_lib.list_envs()] + EXTRA_CASES
    if args.only:
        cases = [case for case in cases if case[0] in args.only]
    records = []
    for env_id, kwargs in cases:
        record = run_case(env_id, kwargs, args.episodes, args.seed)
        records.append(record)
        if not args.markdown:
            base = record["baseline"]
            print(
                f"{label(record):52s} random {fmt(record['random'].mean_return):>9s} "
                f"(+-{fmt(record['random'].ci95):>7s}, {record['random'].mean_length:5.0f} steps)  "
                + (
                    f"baseline {fmt(base.mean_return):>9s} (+-{fmt(base.ci95):>7s}, "
                    f"{base.mean_length:5.0f} steps)"
                    if base is not None
                    else "baseline -"
                )
                + f"  [{record['seconds']:.1f} s]",
                flush=True,
            )
    if args.markdown:
        print("| Environment | Random | Baseline | Episode length (random / baseline) |")
        print("|---|---:|---:|---:|")
        for record in records:
            base = record["baseline"]
            print(
                f"| `{label(record)}` | {fmt(record['random'].mean_return)} | "
                f"{fmt(base.mean_return) if base is not None else '-'} | "
                f"{record['random'].mean_length:.0f} / "
                f"{'-' if base is None else round(base.mean_length)} |"
            )


if __name__ == "__main__":
    main()
