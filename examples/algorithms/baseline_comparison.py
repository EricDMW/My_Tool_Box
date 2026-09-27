"""Compare your own method with the integrated baselines.

This is the comparison a paper needs: your method next to learning baselines
(MAPPO and IPPO with their tuned presets), the environment's classical
controller and random actions -- all evaluated on the same seeded episodes,
with the learning baselines trained over several seeds. One call to
``marl_algorithms.compare`` does it.

"Your method" here is a distributed variant of droop control for
``PowerGrid-v0``: every bus reacts to its own frequency deviation *and* to the
mean deviation of its neighbours,

    u_i = clip(-k * (omega_i + beta * mean_{j in N(i)} omega_j), -u_max_i, u_max_i).

Replace ``my_method`` with your own policy: any function mapping the batched
observations ``(num_envs, n_agents, obs_dim)`` to batched actions (wrap a
single-environment policy with ``marl_algorithms.per_copy``), or a trained
``marl_algorithms`` algorithm.

The script prints the table, saves it as CSV and draws the mean return of
every method with its spread over seeds.

Run::

    python examples/algorithms/baseline_comparison.py                 # 2 seeds, about 6 minutes
    python examples/algorithms/baseline_comparison.py --seeds 0 1 2 --algos mappo ippo maddpg
    python examples/algorithms/baseline_comparison.py --quick         # smoke test, seconds

The same comparison without code, for the learning baselines only::

    marl-train compare PowerGrid-v0 --algos mappo ippo --seeds 0 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from env_lib.power_grid_env import OBSERVATION_FEATURES, OBSERVATION_SCALE
from marl_algorithms import Comparison, compare

COLUMN = {name: index for index, name in enumerate(OBSERVATION_FEATURES)}


def my_method(gain: float = 0.25, neighbour_weight: float = 0.5):
    """Droop control that also reacts to the neighbours' mean frequency deviation."""

    def policy(obs: np.ndarray) -> np.ndarray:
        def feature(name: str) -> np.ndarray:
            return obs[..., COLUMN[name]].astype(np.float64) / OBSERVATION_SCALE[name]

        signal = feature("omega") + neighbour_weight * feature("neighbour_omega_mean")
        capacity = feature("capacity")
        return np.clip(-gain * signal, -capacity, capacity).astype(np.float32)

    return policy


def plot(report: Comparison, path: Path) -> Path:
    """Horizontal bars: mean return of every method, with the spread over seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from toolkit.plotkit import OKABE_ITO_COLOR_LIST, save_figure, style_context

    rows = [row for row in report.rows if row.name != "random"]  # random is off the scale
    labels = {"baseline": "classical controller"}
    colours = {"reference": OKABE_ITO_COLOR_LIST[2], "algorithm": OKABE_ITO_COLOR_LIST[4]}
    with style_context("research"):
        fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(rows) + 1.1))
        y = np.arange(len(rows))[::-1]
        means = np.array([row.mean for row in rows])
        ax.barh(y, means, color=[colours.get(row.kind, OKABE_ITO_COLOR_LIST[5]) for row in rows])
        spread = np.array([row.std for row in rows])
        seeded = ~np.isnan(spread)  # the spread over training seeds, where there is one
        ax.errorbar(
            means[seeded], y[seeded], xerr=spread[seeded], fmt="none", ecolor="0.15", capsize=3
        )
        ax.set_yticks(y)
        ax.set_yticklabels([labels.get(row.name, row.name) for row in rows], fontsize=9)
        ax.axvline(0.0, color="0.3", lw=0.8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("mean team return (higher is better)", fontsize=9)
        random = next((row.mean for row in report.rows if row.name == "random"), None)
        note = f"; random actions {random:.4g}" if random is not None else ""
        ax.set_title(f"{report.env_id}, {report.n_episodes} episodes{note}", fontsize=9)
        fig.tight_layout()
        saved = save_figure(fig, path.with_suffix(""), formats=("png",))
        plt.close(fig)
    return Path(saved[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algos", nargs="+", default=["mappo", "ippo"], help="learning baselines")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1], help="training seeds")
    parser.add_argument("--episodes", type=int, default=64, help="evaluation episodes")
    parser.add_argument("--quick", action="store_true", help="small budgets (smoke test)")
    parser.add_argument("--csv", default="renders/baseline_comparison.csv", help="table as CSV")
    parser.add_argument(
        "--plot", default="renders/baseline_comparison.png", help="figure ('' skips)"
    )
    parser.add_argument("--threads", type=int, default=1, help="PyTorch threads")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    budget = {"total_steps": 4_096, "num_envs": 16} if args.quick else {}
    report = compare(
        "PowerGrid-v0",
        args.algos,
        seeds=args.seeds,
        policies={"my method": my_method()},
        n_episodes=args.episodes,
        verbose=True,
        **budget,
    )
    print()
    print(report)
    best = report.ranking()[0]
    print(f"\nbest learned or proposed method: {best.name} ({best.mean:.4g})")
    print(f"saved {report.to_csv(args.csv)}")
    if args.plot:
        print(f"saved {plot(report, Path(args.plot))}")


if __name__ == "__main__":
    main()
