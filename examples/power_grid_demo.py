"""Power grid frequency control: zero, random and droop control on PowerGrid-v0.

The demo creates ``PowerGrid-v0`` (swing equations of a networked power
system, one agent per bus) through :func:`env_lib.make`, rolls out a few seeded
episodes with three controllers and prints a small comparison table:

* ``zero``   -- no control (only the inherent damping of the buses),
* ``random`` -- uniformly random injections within the bounds,
* ``droop``  -- decentralised droop control ``u_i = -k omega_i`` computed from
  every agent's own observation (:func:`env_lib.power_grid_env.droop_policy`).

All controllers face the same disturbances (the disturbance sequence of a
seeded episode does not depend on the actions). The table reports the mean
episode return, the frequency nadir (lowest bus frequency deviation), the
largest absolute deviation, the settling time (the last time any bus was
outside +/-20 mHz; the episode length means "never settled") and the number
of episodes that tripped the frequency limit. Optionally the droop controller
is recorded as a GIF or MP4.

Examples
--------
Default benchmark system (16 buses, small-world network)::

    python examples/power_grid_demo.py

A larger ring network, five episodes, with a recording in the light theme::

    python examples/power_grid_demo.py --buses 32 --topology ring --episodes 5 \\
        --theme light --render renders/power_grid.gif

Watch the droop controller live in a window::

    python examples/power_grid_demo.py --render-mode human
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np

import env_lib
from env_lib.power_grid_env import DEFAULT_DROOP_GAIN, TOPOLOGIES, droop_policy
from env_lib.utils import record_episode, set_theme

SETTLING_BAND_HZ = 0.02


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--buses", type=int, default=16, help="number of buses (agents)")
    parser.add_argument("--topology", choices=TOPOLOGIES, default="small_world")
    parser.add_argument("--episodes", type=int, default=3, help="seeded episodes per controller")
    parser.add_argument("--steps", type=int, default=200, help="episode length (max_steps)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first episode")
    parser.add_argument("--gain", type=float, default=DEFAULT_DROOP_GAIN, help="droop gain")
    parser.add_argument("--noise", type=float, default=0.0, help="OU load noise std (pu)")
    parser.add_argument(
        "--render",
        "--save",
        dest="render",
        default="",
        metavar="PATH",
        help="record the droop controller to PATH (.gif or .mp4)",
    )
    parser.add_argument("--render-mode", choices=("rgb_array", "human"), default="rgb_array")
    parser.add_argument("--render-every", type=int, default=1, help="record every k-th step")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    return parser.parse_args(argv)


def make_env(args: argparse.Namespace, render_mode: str | None = None):
    return env_lib.make(
        "PowerGrid-v0",
        n_buses=args.buses,
        topology=args.topology,
        max_steps=args.steps,
        noise_std=args.noise,
        render_mode=render_mode,
    )


def run_episode(env, policy: Callable[[np.ndarray], np.ndarray], seed: int) -> dict[str, float]:
    """Roll out one episode and summarise it (frequencies in Hz)."""
    obs, info = env.reset(seed=seed)
    total, last_outside = 0.0, 0.0
    band = 2.0 * math.pi * SETTLING_BAND_HZ
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        if info["max_abs_omega"] > band:
            last_outside = info["time"]
        if terminated or truncated:
            return {
                "return": total,
                "nadir": info["frequency_nadir"] / (2.0 * math.pi),
                "peak": info["peak_abs_omega"] / (2.0 * math.pi),
                "settle": last_outside,
                "tripped": float(info["tripped"]),
            }


def main(argv: list | None = None) -> None:
    args = parse_args(argv)
    set_theme(args.theme)

    env = make_env(args)
    core = env.unwrapped
    print(
        f"PowerGrid-v0: {core.n_buses} buses, topology={args.topology}, "
        f"{core.edges.shape[0]} lines, horizon {args.steps * core.dt:.1f} s, "
        f"trip limit +/-{core.config.frequency_limit:g} Hz"
    )
    rng = np.random.default_rng(args.seed)
    u_max = core.u_max
    policies: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "zero": lambda obs: np.zeros(core.n_buses, dtype=np.float32),
        "random": lambda obs: rng.uniform(-u_max, u_max).astype(np.float32),
        "droop": lambda obs: droop_policy(obs, gain=args.gain),
    }
    seeds = range(args.seed, args.seed + args.episodes)
    print(
        f"{'policy':<8} {'return':>10} {'(std)':>9} {'nadir [Hz]':>11} {'max dev [Hz]':>13} "
        f"{'settle [s]':>11} {'trips':>6}"
    )
    for name, policy in policies.items():
        results = [run_episode(env, policy, seed) for seed in seeds]
        returns = np.array([r["return"] for r in results])
        mean = {key: float(np.mean([r[key] for r in results])) for key in results[0]}
        trips = sum(int(r["tripped"]) for r in results)
        print(
            f"{name:<8} {returns.mean():>10.2f} {returns.std():>9.2f} {mean['nadir']:>11.3f} "
            f"{mean['peak']:>13.3f} {mean['settle']:>11.2f} {trips:>6d}"
        )
    env.close()

    if args.render_mode == "human":
        live = make_env(args, render_mode="human")
        run_episode(live, lambda obs: droop_policy(obs, gain=args.gain), args.seed)
        live.close()
        return

    if args.render:
        recorder = make_env(args, render_mode="rgb_array")
        frames = record_episode(
            recorder,
            policy=lambda obs: droop_policy(obs, gain=args.gain),
            path=Path(args.render),
            seed=args.seed,
            render_every=max(1, args.render_every),
        )
        recorder.close()
        print(f"saved {len(frames)} frames of the droop controller to {args.render}")


if __name__ == "__main__":
    main()
