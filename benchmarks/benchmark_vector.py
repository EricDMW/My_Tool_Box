"""Throughput of native batched vector environments versus ``SyncVectorEnv``.

For every environment id and number of copies, the script creates
``env_lib.make_vec(id, num_envs)`` (the native :class:`BatchedVectorEnv`
implementation when the id registers one) and
``env_lib.make_vec(id, num_envs, vectorization_mode="sync")`` (Gymnasium's
``SyncVectorEnv`` over single environments), steps both with random actions
(automatic resets included) and reports

* env-steps/s -- copies advanced per second (``num_envs * steps / time``),
* agent-steps/s -- env-steps/s times the number of agents of one copy (rows of a
  joint ``(n_agents, obs_dim)`` observation; oscillators for Kuramoto),
* the speed-up of the native implementation.

Environments that cannot be imported or created (e.g. optional or unfinished
ones) are skipped with a note. ``SyncVectorEnv`` is skipped above
``--sync-max`` copies because it scales linearly and gets slow.

Run::

    python benchmarks/benchmark_vector.py
    python benchmarks/benchmark_vector.py --quick            # CI smoke run
    python benchmarks/benchmark_vector.py --ids Consensus-v0 --num-envs 64 1024 --markdown

For reproducible numbers pin the BLAS threads (``OMP_NUM_THREADS=1``).
"""

from __future__ import annotations

import argparse
import os
import platform
import time
import warnings
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy

import env_lib

DEFAULT_IDS = (
    "Consensus-v0",
    "Formation-v0",
    "KuramotoOscillator-v0",
    "PowerGrid-v0",
    "Platoon-v0",
)
DEFAULT_NUM_ENVS = (1, 16, 64, 256, 1024)


def agents_per_env(envs: Any) -> int:
    """Agents of one copy: rows of a joint observation, else the oscillators."""
    shape = envs.single_observation_space.shape
    if shape is not None and len(shape) == 2:
        return int(shape[0])
    unwrapped = getattr(envs, "unwrapped", envs)
    for name in ("n_agents", "n_oscillators"):
        value = getattr(unwrapped, name, None)
        if isinstance(value, int) and value > 1:
            return value
    return 1


def throughput(envs: Any, min_steps: int, env_steps: int, seed: int) -> tuple[float, int]:
    """Environment steps per second (random actions, autoreset on).

    Runs at least ``min_steps`` batched steps and at least ``env_steps`` single
    environment steps; returns ``(env_steps_per_second, batched_steps)``.
    """
    envs.reset(seed=seed)
    envs.action_space.seed(seed)
    pool = [envs.action_space.sample() for _ in range(8)]
    for action in pool[:3]:  # warm-up (lazy allocations, first autoresets)
        envs.step(action)
    steps = max(min_steps, -(-env_steps // envs.num_envs))
    start = time.perf_counter()
    for k in range(steps):
        envs.step(pool[k % len(pool)])
    elapsed = time.perf_counter() - start
    return envs.num_envs * steps / elapsed, steps


def benchmark(
    ids: list[str], num_envs: list[int], sync_max: int, min_steps: int, env_steps: int, seed: int
) -> list[dict]:
    rows = []
    for env_id in ids:
        try:
            spec = env_lib.get_spec(env_id)
            probe = env_lib.make_vec(env_id, 1)
            agents = agents_per_env(probe)
            probe.close()
        except Exception as exc:  # unavailable or unfinished environment: skip it
            rows.append({"id": env_id, "note": f"skipped ({type(exc).__name__}: {exc})"})
            continue
        native_available = spec.vector_entry_point is not None
        for count in num_envs:
            row: dict[str, Any] = {"id": env_id, "num_envs": count, "agents": agents, "note": ""}
            if native_available:
                envs = env_lib.make_vec(env_id, count)
                row["native_class"] = type(envs).__name__
                row["native"], row["native_steps"] = throughput(envs, min_steps, env_steps, seed)
                envs.close()
            else:
                row["note"] = "no native implementation"
            if count <= sync_max:
                envs = env_lib.make_vec(env_id, count, vectorization_mode="sync")
                row["sync"], row["sync_steps"] = throughput(envs, min_steps, env_steps, seed)
                envs.close()
            rows.append(row)
    return rows


def _fmt(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def report(rows: list[dict], markdown: bool) -> None:
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    print(
        f"env_lib {env_lib.__version__} | Python {platform.python_version()} | "
        f"NumPy {numpy.__version__} | {platform.machine()} | {os.cpu_count()} CPUs | "
        f"OMP_NUM_THREADS={threads}"
    )
    header = (
        "environment",
        "num_envs",
        "native env-steps/s",
        "sync env-steps/s",
        "speed-up",
        "native agent-steps/s",
    )
    lines = []
    for row in rows:
        if "num_envs" not in row:
            lines.append((row["id"], "-", "-", "-", "-", row["note"]))
            continue
        native, sync = row.get("native"), row.get("sync")
        speedup = f"{native / sync:.1f}x" if native and sync else "-"
        agent_steps = native * row["agents"] if native else None
        lines.append(
            (
                row["id"],
                str(row["num_envs"]),
                _fmt(native),
                _fmt(sync) if sync else "skipped" if native else "-",
                speedup,
                _fmt(agent_steps) + (f" {row['note']}" if row["note"] else ""),
            )
        )
    if markdown:
        print("\n| " + " | ".join(header) + " |")
        print("|---|---:|---:|---:|---:|---:|")
        for line in lines:
            print("| " + " | ".join(line) + " |")
        return
    widths = [max(len(str(item)) for item in column) for column in zip(header, *lines)]
    print()
    print(
        "  ".join(
            f"{item:>{w}}" if i else f"{item:<{w}}"
            for i, (item, w) in enumerate(zip(header, widths))
        )
    )
    for line in lines:
        print(
            "  ".join(
                f"{item:>{w}}" if i else f"{item:<{w}}"
                for i, (item, w) in enumerate(zip(line, widths))
            )
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS), help="environment ids")
    parser.add_argument(
        "--num-envs", nargs="+", type=int, default=list(DEFAULT_NUM_ENVS), help="batch sizes"
    )
    parser.add_argument(
        "--sync-max", type=int, default=256, help="largest num_envs timed with SyncVectorEnv"
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=300,
        help="minimum batched steps (300 spans a full episode, so autoresets are included)",
    )
    parser.add_argument(
        "--env-steps", type=int, default=20000, help="minimum single-environment steps per case"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="small, fast run (CI smoke test)")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args(argv)
    if args.quick:
        args.num_envs = [n for n in args.num_envs if n <= 64] or [1, 16, 64]
        args.sync_max = min(args.sync_max, 64)
        args.min_steps, args.env_steps = 5, 500
    warnings.simplefilter("ignore")
    rows = benchmark(
        args.ids, args.num_envs, args.sync_max, args.min_steps, args.env_steps, args.seed
    )
    report(rows, args.markdown)


if __name__ == "__main__":
    main()
