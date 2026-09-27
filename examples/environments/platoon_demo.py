"""Vehicle platoon control: random actions vs. ACC vs. cooperative ACC (CACC).

The demo creates ``Platoon-v0`` through :func:`env_lib.make` and rolls out a few
seeded episodes with three decentralised policies:

* ``random`` -- uniformly random accelerations;
* ``ACC`` -- sensor-only adaptive cruise control,
  ``u_i = k_p e_i + k_d (v_{i-1} - v_i)`` (:func:`cacc_policy` with ``k_a=0``);
* ``CACC`` -- the same law plus the communicated acceleration of the
  predecessor, ``+ k_a a_{i-1}`` (:func:`env_lib.platoon_env.cacc_policy`).

It prints the episode return, the number of collisions, the smallest gap and
the peak spacing error of the first and the last follower. With the default
short time headway (0.6 s) ACC is string unstable -- spacing errors grow from
vehicle to vehicle and the platoon collides in stop-and-go traffic -- while
CACC attenuates them. The analytic peak gain of the string-stability transfer
function (:func:`string_stability_gain`, string stable when it is 1) is
printed for both controllers. ``--render`` records the CACC controller.

Examples
--------
Default comparison (8 followers, predecessor following, mixed scenario)::

    python examples/environments/platoon_demo.py

Stop-and-go wave, predecessor-leader topology, GIF of the CACC controller::

    python examples/environments/platoon_demo.py --scenario stop_and_go --topology predecessor_leader \\
        --render renders/platoon.gif

ACC with a longer headway is string stable again::

    python examples/environments/platoon_demo.py --headway 1.2 --scenario stop_and_go

Watch the controller live in a window::

    python examples/environments/platoon_demo.py --render-mode human
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

import env_lib
from env_lib.platoon_env import (
    CACC_GAINS,
    SCENARIOS,
    TOPOLOGIES,
    cacc_policy,
    string_stability_gain,
)
from env_lib.utils import record_episode, set_theme

Policy = Callable[[np.ndarray], np.ndarray]


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vehicles", type=int, default=8, help="number of followers (agents)")
    parser.add_argument("--topology", choices=TOPOLOGIES, default="predecessor")
    parser.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    parser.add_argument("--headway", type=float, default=0.6, help="time headway h in seconds")
    parser.add_argument("--steps", type=int, default=600, help="episode length (max_steps)")
    parser.add_argument("--episodes", type=int, default=5, help="seeded episodes per policy")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first episode")
    parser.add_argument(
        "--render",
        default="",
        metavar="PATH",
        help="record CACC as GIF/MP4 (e.g. renders/platoon.gif)",
    )
    parser.add_argument("--render-mode", choices=("rgb_array", "human"), default="rgb_array")
    parser.add_argument("--render-every", type=int, default=2, help="record every k-th step")
    parser.add_argument("--fps", type=float, default=20.0, help="playback rate of the recording")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    return parser.parse_args(argv)


def make_env(args: argparse.Namespace, render_mode: str | None = None):
    return env_lib.make(
        "Platoon-v0",
        n_followers=args.vehicles,
        topology=args.topology,
        scenario=args.scenario,
        headway=args.headway,
        max_steps=args.steps,
        render_mode=render_mode,
    )


def run_episode(env, policy: Policy, seed: int) -> dict[str, float]:
    """Roll out one episode and summarise it."""
    obs, info = env.reset(seed=seed)
    total, min_gap = 0.0, info["min_gap"]
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        min_gap = min(min_gap, info["min_gap"])
        if terminated or truncated:
            peaks = info["peak_spacing_errors"]
            return {
                "return": total,
                "collision": float(info["collision"]),
                "min_gap": min_gap,
                "first": float(peaks[0]),
                "last": float(peaks[-1]),
                "steps": float(info["step"]),
            }


def main(argv: list | None = None) -> None:
    args = parse_args(argv)
    # Unbounded observations and physical (asymmetric) action bounds are deliberate.
    warnings.filterwarnings("ignore", message=r".*Box observation space m\w+ value is")
    warnings.filterwarnings("ignore", message=r".*symmetric and normalized space")
    set_theme(args.theme)

    env = make_env(args)
    base = env.unwrapped
    low, high = base.config.tau_range
    taus = np.linspace(low, high, 9)  # worst case over the range of actuator lags
    gains = {
        name: max(
            string_stability_gain(k_a=k_a, headway=base.headway, time_constant=tau, dt=base.dt)
            for tau in taus
        )
        for name, k_a in (("ACC", 0.0), ("CACC", CACC_GAINS["k_a"]))
    }
    print(
        f"Platoon demo: {base.n_followers} followers, topology={base.topology}, "
        f"scenario={base.scenario}, headway={base.headway:g} s, lags in [{low:.2f}, {high:.2f}] s"
    )
    print(
        f"string-stability gain max|Gamma| (1 = string stable): "
        f"ACC {gains['ACC']:.3f}, CACC {gains['CACC']:.3f}"
    )

    rng = np.random.default_rng(args.seed)
    space = base.action_space
    policies: dict[str, Policy] = {
        "random": lambda _obs: rng.uniform(space.low, space.high).astype(np.float32),
        "ACC": lambda obs: cacc_policy(obs, k_a=0.0),
        "CACC": cacc_policy,
    }
    seeds = range(args.seed, args.seed + args.episodes)
    header = (
        f"{'policy':<8} {'return':>10} {'collisions':>11} {'min gap [m]':>12} "
        f"{'peak|e_1| [m]':>14} {'peak|e_n| [m]':>14} {'steps':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, policy in policies.items():
        runs = [run_episode(env, policy, seed) for seed in seeds]
        mean = {key: float(np.mean([run[key] for run in runs])) for key in runs[0]}
        collisions = int(sum(run["collision"] for run in runs))
        min_gap = min(run["min_gap"] for run in runs)
        print(
            f"{name:<8} {mean['return']:>10.1f} {collisions:>7d}/{len(runs):<3d} {min_gap:>12.2f} "
            f"{mean['first']:>14.2f} {mean['last']:>14.2f} {mean['steps']:>6.0f}"
        )
    print("(means over episodes; min gap is the smallest over all episodes)")
    env.close()

    if args.render_mode == "human":
        live = make_env(args, render_mode="human")
        run_episode(live, cacc_policy, args.seed)
        live.close()
        return

    if args.render:
        recorder = make_env(args, render_mode="rgb_array")
        frames = record_episode(
            recorder,
            policy=cacc_policy,
            path=Path(args.render),
            seed=args.seed,
            fps=args.fps,
            render_every=max(1, args.render_every),
        )
        recorder.close()
        print(f"saved {len(frames)} frames of the CACC controller to {args.render}")


if __name__ == "__main__":
    main()
