"""Train classical multi-agent RL algorithms on the env_lib environments.

The default suite trains one algorithm of every family on a different
environment, with the tuned presets of ``marl_algorithms`` (each run takes one
to two minutes on one CPU core):

* MAPPO  on ``PowerGrid-v0``     -- continuous actions, 16 agents, on-policy;
* MADDPG on ``Consensus-v0``     -- continuous actions, 8 agents, off-policy;
* QMIX   on ``LineMsg-v0``       -- discrete actions, 10 agents, value-based;
* VDN    on ``WirelessComm-v1``  -- discrete actions, 16 agents, value-based.

Each run trains on a vector environment of parallel copies
(``marl_algorithms.make_vector_env``: one native batch for the continuous
environments, ``gymnasium.vector.SyncVectorEnv`` for LineMsg and
WirelessComm), then evaluates random actions, the
trained policy and the environment's classical controller
(``env_lib.baseline_policy``) on the same seeded episodes. The script prints a
results table and saves the learning curves.

Run::

    python examples/marl_training_demo.py                          # default suite
    python examples/marl_training_demo.py --algo mappo --env PowerGrid-v0
    python examples/marl_training_demo.py --algo qmix --env LineMsg-v0 --gif renders/qmix.gif
    python examples/marl_training_demo.py --quick                  # smoke test, seconds
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import env_lib
from env_lib.utils import record_episode, set_theme
from env_lib.utils.evaluation import evaluate
from marl_algorithms import make_vector_env
from marl_algorithms.presets import get_preset, train_preset

SUITE: list[tuple[str, str]] = [
    ("mappo", "PowerGrid-v0"),
    ("maddpg", "Consensus-v0"),
    ("qmix", "LineMsg-v0"),
    ("vdn", "WirelessComm-v1"),
]


@dataclass
class RunResult:
    algorithm: str
    env_id: str
    env_steps: int
    seconds: float
    random: float
    trained: float
    baseline: float | None
    curve: tuple[np.ndarray, np.ndarray]


def run(
    algorithm: str, env_id: str, *, quick: bool, seed: int, episodes: int, gif: str | None
) -> RunResult:
    """Train one preset and evaluate it against random actions and the baseline."""
    preset = get_preset(algorithm, env_id)
    if preset is None:
        raise SystemExit(f"no preset for {algorithm} on {env_id}; run `marl-train presets`")
    overrides: dict[str, Any] = {}
    if quick:  # smoke test: a few updates only
        overrides = {
            "total_steps": 2_000 if algorithm in ("mappo", "ippo") else 1_500,
            "num_envs": 8,
        }
        if algorithm not in ("mappo", "ippo"):
            overrides["warmup_steps"] = 500
    start = time.perf_counter()
    algo, log = train_preset(algorithm, env_id, seed=seed, **overrides)
    seconds = time.perf_counter() - start

    env_kwargs = preset.get("env_kwargs", {})
    envs = make_vector_env(env_id, min(episodes, 64), **env_kwargs)
    random_result = evaluate(envs, None, n_episodes=episodes, seed=seed + 1)
    trained_result = algo.evaluate(envs, episodes, seed=seed + 1)
    try:
        baseline_result = evaluate(
            envs, env_lib.baseline_policy(envs), n_episodes=episodes, seed=seed + 1
        )
        baseline = baseline_result.mean_return
    except (TypeError, NotImplementedError):
        baseline = None
    envs.close()

    if gif:
        env = env_lib.make(env_id, render_mode="rgb_array", **env_kwargs)
        frames = record_episode(env, algo.policy(), gif, seed=seed + 1, render_every=2)
        env.close()
        print(f"  saved {len(frames)} frames of the trained {algorithm} policy to {gif}")

    return RunResult(
        algorithm=algorithm,
        env_id=env_id,
        env_steps=algo.env_steps,
        seconds=seconds,
        random=random_result.mean_return,
        trained=trained_result.mean_return,
        baseline=baseline,
        curve=log.curve(points=40),
    )


def print_table(results: list[RunResult], episodes: int) -> None:
    print(f"\nMean episode return over {episodes} seeded evaluation episodes (higher is better):\n")
    header = f"{'algorithm':10s}{'environment':18s}{'env steps':>11s}{'time [s]':>10s}"
    header += f"{'random':>12s}{'trained':>12s}{'baseline':>12s}"
    print(header)
    print("-" * len(header))
    for r in results:
        baseline = f"{r.baseline:12.4g}" if r.baseline is not None else f"{'-':>12s}"
        print(
            f"{r.algorithm:10s}{r.env_id:18s}{r.env_steps:>11,d}{r.seconds:>10.1f}"
            f"{r.random:>12.4g}{r.trained:>12.4g}{baseline}"
        )


def plot_curves(results: list[RunResult], path: Path) -> Path:
    """Training-return curves with the random and baseline evaluation returns as reference lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from toolkit.plotkit import OKABE_ITO_COLOR_LIST, save_figure, style_context

    thousands = FuncFormatter(lambda x, _: f"{x / 1000:.0f}k" if x else "0")
    with style_context("research"):
        fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 2.7), squeeze=False)
        for ax, r in zip(axes[0], results):
            steps, returns = r.curve
            ax.plot(
                steps, returns, color=OKABE_ITO_COLOR_LIST[4], lw=1.8, label=r.algorithm.upper()
            )
            ax.axhline(r.random, color=OKABE_ITO_COLOR_LIST[0], ls=":", lw=1.2, label="random")
            if r.baseline is not None:
                ax.axhline(
                    r.baseline, color=OKABE_ITO_COLOR_LIST[2], ls="--", lw=1.2, label="baseline"
                )
            ax.set_title(f"{r.algorithm.upper()} on {r.env_id}", fontsize=10)
            ax.xaxis.set_major_formatter(thousands)
            ax.tick_params(labelsize=8)
            ax.set_xlabel("environment steps", fontsize=9)
            ax.set_ylabel("training return", fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout()
        saved = save_figure(fig, path.with_suffix(""), formats=("png",))
        plt.close(fig)
    return Path(saved[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algo", help="train only this algorithm (with --env)")
    parser.add_argument("--env", help="environment id for --algo")
    parser.add_argument("--quick", action="store_true", help="a few updates per run (smoke test)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=64, help="evaluation episodes")
    parser.add_argument("--plot", default="renders/marl_training.png", help="learning-curve figure")
    parser.add_argument("--gif", help="record the trained policy of a single run (--algo/--env)")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch threads")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    set_theme("light")
    if args.algo or args.env:
        if not (args.algo and args.env):
            raise SystemExit("--algo and --env go together")
        suite = [(args.algo.lower(), args.env)]
    else:
        suite = SUITE
    results = []
    for algorithm, env_id in suite:
        print(f"Training {algorithm.upper()} on {env_id} ...", flush=True)
        results.append(
            run(
                algorithm,
                env_id,
                quick=args.quick,
                seed=args.seed,
                episodes=args.episodes,
                gif=args.gif if len(suite) == 1 else None,
            )
        )
        r = results[-1]
        print(f"  {r.env_steps:,} env steps in {r.seconds:.1f} s; trained return {r.trained:.4g}")
    print_table(results, args.episodes)
    if args.plot:
        print(f"\nlearning curves: {plot_curves(results, Path(args.plot))}")


if __name__ == "__main__":
    main()
