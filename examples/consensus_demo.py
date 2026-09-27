"""Networked consensus and formation control: random policy vs. Laplacian feedback.

The demo creates ``Consensus-v0`` (or ``Formation-v0``) through :func:`env_lib.make`,
rolls out one episode with uniformly random actions and one with the classic
distributed controller ``u_i = -gain * sum_{j in N_i} ((x_i - d_i) - (x_j - d_j))``
(:meth:`ConsensusEnv.laplacian_policy`), prints the final task error, the step at
which the task was solved and the episode return, and records the Laplacian
controller as a GIF.

Examples
--------
Rendezvous on a ring (default)::

    python examples/consensus_demo.py

Wedge formation of double integrators on a proximity graph, light theme::

    python examples/consensus_demo.py --task formation --shape wedge \\
        --topology proximity --dynamics double --theme light --save renders/wedge.gif

Watch the controller live in a window::

    python examples/consensus_demo.py --render-mode human
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

import env_lib
from env_lib.consensus_env import DYNAMICS, FORMATION_SHAPES, TASKS, TOPOLOGIES
from env_lib.utils import record_episode, set_theme


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=TASKS, default="consensus")
    parser.add_argument(
        "--shape", choices=FORMATION_SHAPES, default="circle", help="formation shape"
    )
    parser.add_argument("--topology", choices=TOPOLOGIES, default="ring")
    parser.add_argument("--dynamics", choices=DYNAMICS, default="single")
    parser.add_argument("--n-agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200, help="episode length (max_steps)")
    parser.add_argument("--gain", type=float, default=1.0, help="Laplacian feedback gain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save",
        default="renders/consensus.gif",
        help="GIF/MP4 of the Laplacian controller ('' to skip)",
    )
    parser.add_argument("--render-mode", choices=("rgb_array", "human"), default="rgb_array")
    parser.add_argument("--render-every", type=int, default=1, help="record every k-th step")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    return parser.parse_args(argv)


def make_env(args: argparse.Namespace, render_mode: str | None = None):
    env_id = "Formation-v0" if args.task == "formation" else "Consensus-v0"
    return env_lib.make(
        env_id,
        task=args.task,
        formation_shape=args.shape,
        topology=args.topology,
        dynamics=args.dynamics,
        n_agents=args.n_agents,
        max_steps=args.steps,
        graph_seed=args.seed,
        render_mode=render_mode,
    )


def run_episode(env, policy: Callable[[], np.ndarray], seed: int) -> dict[str, float]:
    """Roll out one episode and summarise it."""
    env.reset(seed=seed)
    total, solved_at = 0.0, None
    while True:
        _, reward, terminated, truncated, info = env.step(policy())
        total += reward
        if info["success"] and solved_at is None:
            solved_at = info["step"]
        if terminated or truncated:
            return {
                "error": info["error"],
                "steps": info["step"],
                "solved_at": solved_at,
                "return": total,
                "lambda2": info["algebraic_connectivity"],
            }


def main(argv: list | None = None) -> None:
    args = parse_args(argv)
    # The observation space is intentionally unbounded; silence Gymnasium's generic hint.
    warnings.filterwarnings("ignore", message=r".*Box observation space m\w+ value is")
    set_theme(args.theme)

    env = make_env(args)
    base = env.unwrapped
    task_label = f"formation({args.shape})" if args.task == "formation" else "consensus"
    print(
        f"Consensus demo: task={task_label}, topology={args.topology}, "
        f"dynamics={args.dynamics}, N={args.n_agents}, seed={args.seed}"
    )
    _, info = env.reset(seed=args.seed)
    print(
        f"graph: lambda2={info['algebraic_connectivity']:.3f} at reset, "
        f"observation {base.observation_space.shape}, tolerance={base.tolerance}"
    )

    env.action_space.seed(args.seed)
    policies = {
        "random": env.action_space.sample,
        "laplacian": lambda: base.laplacian_policy(args.gain),
    }
    print(f"{'policy':<10} {'final error':>12} {'solved at':>10} {'return':>10}")
    for name, policy in policies.items():
        result = run_episode(env, policy, args.seed)
        solved = "-" if result["solved_at"] is None else str(result["solved_at"])
        print(f"{name:<10} {result['error']:>12.4f} {solved:>10} {result['return']:>10.1f}")
    env.close()

    if args.render_mode == "human":
        live = make_env(args, render_mode="human")
        run_episode(live, lambda: live.unwrapped.laplacian_policy(args.gain), args.seed)
        live.close()
        return

    if args.save:
        recorder = make_env(args, render_mode="rgb_array")
        frames = record_episode(
            recorder,
            policy=lambda _obs: recorder.unwrapped.laplacian_policy(args.gain),
            path=Path(args.save),
            seed=args.seed,
            render_every=max(1, args.render_every),
        )
        recorder.close()
        print(f"saved {len(frames)} frames to {args.save}")


if __name__ == "__main__":
    main()
