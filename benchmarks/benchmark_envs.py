"""Micro-benchmarks for the env_lib environments.

Measures the mean wall-clock time of ``env.step`` (random actions, no
rendering) and of ``env.render`` in ``"rgb_array"`` mode for every
environment, then prints a table. Optional dependencies (torch, pygame,
pymunk) are skipped when missing.

Run::

    python benchmarks/benchmark_envs.py
    python benchmarks/benchmark_envs.py --steps 500 --only kuramoto ajlatt
    python benchmarks/benchmark_envs.py --markdown > renders/benchmarks.md
"""

from __future__ import annotations

import argparse
import os
import platform
import time
import warnings
from typing import Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

import env_lib

Factory = Callable[..., object]


def _cases() -> dict[str, list[tuple]]:
    """Benchmark cases: group -> [(label, factory, kwargs), ...]."""
    return {
        "kuramoto": [
            ("Kuramoto NumPy, N=10", env_lib.KuramotoOscillatorEnv, {"n_oscillators": 10}),
            ("Kuramoto NumPy, N=50", env_lib.KuramotoOscillatorEnv, {"n_oscillators": 50}),
            (
                "Kuramoto NumPy RK4, N=50",
                env_lib.KuramotoOscillatorEnv,
                {"n_oscillators": 50, "integration_method": "rk4"},
            ),
            (
                "Kuramoto torch, N=50 x 8 systems",
                lambda **kw: env_lib.KuramotoOscillatorEnvTorch(**kw),
                {"n_oscillators": 50, "n_agents": 8},
            ),
        ],
        "linemsg": [("LineMsg, 10 agents", env_lib.LineMsgEnv, {"num_agents": 10})],
        "wireless": [
            ("WirelessComm, 6x6", env_lib.WirelessCommEnv, {"grid_x": 6, "grid_y": 6}),
            ("WirelessComm, 12x12", env_lib.WirelessCommEnv, {"grid_x": 12, "grid_y": 12}),
        ],
        "pistonball": [
            ("Pistonball, 20 pistons", lambda **kw: env_lib.PistonballEnv(**kw), {"n_pistons": 20})
        ],
        "consensus": [
            ("Consensus, 8 agents", env_lib.ConsensusEnv, {}),
            (
                "Formation, 32 agents, proximity",
                env_lib.ConsensusEnv,
                {
                    "task": "formation",
                    "n_agents": 32,
                    "topology": "proximity",
                    "formation_shape": "grid",
                },
            ),
        ],
        "ajlatt": [
            ("AJLATT obstacles04, 4 robots", env_lib.AJLATTEnv, {"map_name": "obstacles04"}),
            ("AJLATT obstacles05, 4 robots", env_lib.AJLATTEnv, {"map_name": "obstacles05"}),
        ],
    }


def _time_steps(env, steps: int, seed: int, render: bool) -> float:
    env.reset(seed=seed)
    env.action_space.seed(seed)
    actions = [env.action_space.sample() for _ in range(steps)]
    elapsed = 0.0
    for action in actions:
        start = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(action)
        if render:
            env.render()
        elapsed += time.perf_counter() - start
        if np.any(terminated) or truncated:
            env.reset()
    return 1e3 * elapsed / steps


def benchmark(only: list[str] | None, steps: int, render_steps: int, seed: int) -> list[dict]:
    rows = []
    for group, cases in _cases().items():
        if only and group not in only:
            continue
        for label, factory, kwargs in cases:
            try:
                env = factory(**kwargs)
            except ImportError as exc:
                rows.append(
                    {"case": label, "step": None, "render": None, "note": f"skipped ({exc.name})"}
                )
                continue
            step_ms = _time_steps(env, steps, seed, render=False)
            env.close()
            render_ms = None
            if render_steps:
                env = factory(render_mode="rgb_array", **kwargs)
                env.reset(seed=seed)
                env.render()  # build the figure outside the timed region
                render_ms = _time_steps(env, render_steps, seed, render=True) - step_ms
                env.close()
            rows.append({"case": label, "step": step_ms, "render": render_ms, "note": ""})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=300, help="timed steps per case")
    parser.add_argument(
        "--render-steps", type=int, default=40, help="timed rendered steps (0 disables)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", help="subset of: " + ", ".join(_cases()))
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    rows = benchmark(args.only, args.steps, args.render_steps, args.seed)

    def fmt(value):
        return "-" if value is None else f"{value:.3f}"

    print(
        f"env_lib {env_lib.__version__} | Python {platform.python_version()} | {platform.machine()}"
    )
    if args.markdown:
        print("\n| Environment | step [ms] | rgb_array frame [ms] |\n|---|---:|---:|")
        for r in rows:
            print(f"| {r['case']} | {fmt(r['step'])} | {fmt(r['render'])} {r['note']}|")
        return
    print(f"\n{'environment':<36} {'step [ms]':>10} {'frame [ms]':>11}")
    for r in rows:
        print(f"{r['case']:<36} {fmt(r['step']):>10} {fmt(r['render']):>11} {r['note']}")


if __name__ == "__main__":
    main()
