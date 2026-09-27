"""Train every marl_algorithms preset and compare it with random actions and the baseline.

For each ``(algorithm, environment)`` preset the script trains with
``marl_algorithms.presets.train_preset`` (seed 0), then evaluates random
actions, the trained deterministic policy and the environment's classical
controller (``env_lib.baseline_policy``) on the same seeded episodes of a
64-copy vector environment. Results are appended as JSON lines to ``--out`` so
that shards can run in parallel processes; ``--report`` prints the table.

Run::

    python benchmarks/benchmark_marl.py --out renders/marl.jsonl
    python benchmarks/benchmark_marl.py --only mappo:PowerGrid-v0 qmix:LineMsg-v0
    taskset -c 0 python benchmarks/benchmark_marl.py --shard 0/3 --out renders/marl.jsonl &
    taskset -c 1 python benchmarks/benchmark_marl.py --shard 1/3 --out renders/marl.jsonl &
    taskset -c 2 python benchmarks/benchmark_marl.py --shard 2/3 --out renders/marl.jsonl &
    python benchmarks/benchmark_marl.py --report renders/marl.jsonl --markdown
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch

import env_lib
from env_lib.utils.evaluation import evaluate
from marl_algorithms import make_vector_env
from marl_algorithms.presets import get_preset, list_presets, train_preset

EPISODES = 64
EVAL_SEED = 1


def run_one(algorithm: str, env_id: str) -> dict:
    """Train one preset and evaluate it; returns a result record."""
    preset = get_preset(algorithm, env_id)
    start = time.perf_counter()
    cpu_start = time.process_time()
    algo, _ = train_preset(algorithm, env_id, seed=0)
    seconds = time.perf_counter() - start
    cpu_seconds = time.process_time() - cpu_start
    envs = make_vector_env(env_id, EPISODES, **preset.get("env_kwargs", {}))
    record = {
        "algorithm": algorithm,
        "env_id": env_id,
        "env_steps": int(algo.env_steps),
        "seconds": round(seconds, 1),
        "cpu_seconds": round(cpu_seconds, 1),
        "random": evaluate(envs, None, n_episodes=EPISODES, seed=EVAL_SEED).mean_return,
        "trained": algo.evaluate(envs, EPISODES, seed=EVAL_SEED).mean_return,
    }
    try:
        baseline = env_lib.baseline_policy(envs)
        record["baseline"] = evaluate(
            envs, baseline, n_episodes=EPISODES, seed=EVAL_SEED
        ).mean_return
    except (TypeError, NotImplementedError):
        record["baseline"] = None
    envs.close()
    return record


def report(path: Path, markdown: bool) -> None:
    """Print the results stored in ``path`` in preset order."""
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[(record["algorithm"], record["env_id"])] = record
    order = [pair for pair in list_presets() if pair in records]

    def fmt(value):
        return "-" if value is None else f"{value:.4g}"

    if markdown:
        print("| Algorithm | Environment | Env steps | Time [s] | Random | Trained | Baseline |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for key in order:
            r = records[key]
            print(
                f"| {r['algorithm'].upper()} | `{r['env_id']}` | {r['env_steps']:,} | {r['cpu_seconds']:.0f} "
                f"| {fmt(r['random'])} | {fmt(r['trained'])} | {fmt(r['baseline'])} |"
            )
        return
    print(
        f"{'algorithm':10s}{'environment':18s}{'env steps':>11s}{'cpu s':>8s}{'random':>11s}{'trained':>11s}{'baseline':>11s}"
    )
    for key in order:
        r = records[key]
        print(
            f"{r['algorithm']:10s}{r['env_id']:18s}{r['env_steps']:>11,d}{r['cpu_seconds']:>8.0f}"
            f"{fmt(r['random']):>11s}{fmt(r['trained']):>11s}{fmt(r['baseline']):>11s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", metavar="ALGO:ENV", help="subset of presets")
    parser.add_argument(
        "--shard", default="0/1", metavar="K/N", help="run every N-th preset starting at K"
    )
    parser.add_argument(
        "--out", default="renders/marl_benchmark.jsonl", help="JSON-lines result file"
    )
    parser.add_argument(
        "--report", metavar="FILE", help="print the table of a result file and exit"
    )
    parser.add_argument("--markdown", action="store_true", help="Markdown table")
    args = parser.parse_args()
    if args.report:
        report(Path(args.report), args.markdown)
        return
    torch.set_num_threads(1)
    pairs = list_presets()
    if args.only:
        wanted = {tuple(item.split(":", 1)) for item in args.only}
        pairs = [pair for pair in pairs if pair in wanted]
    k, n = (int(x) for x in args.shard.split("/"))
    pairs = pairs[k::n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for algorithm, env_id in pairs:
        record = run_one(algorithm, env_id)
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"{algorithm:7s} {env_id:18s} {record['env_steps']:>9,d} steps {record['cpu_seconds']:6.1f} s cpu  "
            f"random {record['random']:.4g}  trained {record['trained']:.4g}  baseline {record['baseline']}",
            flush=True,
        )
    report(out, args.markdown)


if __name__ == "__main__":
    main()
