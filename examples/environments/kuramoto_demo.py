"""Drive a Kuramoto oscillator network to synchronisation and record it.

The demo runs a simple hand-written controller on the Kuramoto environment:

* In dynamic coupling mode it ramps every coupling strength linearly from the
  lower to the upper end of ``coupling_range`` over ``--ramp-steps`` steps, so
  the recording shows the classical transition from incoherent drift to a
  phase-locked cluster.
* In every mode it adds a proportional phase-alignment input
  ``a_i = gain * r * sin(psi - theta_i)`` (clipped to the control range), where
  ``r exp(i psi)`` is the order parameter computed from the observation.

The episode ends when the order parameter exceeds the synchronisation
threshold (``terminated``) or after ``--steps`` steps. With the default
``--render-mode rgb_array`` the frames are written to ``renders/kuramoto.gif``
(``.mp4`` also works when ``imageio-ffmpeg`` is installed).

Examples
--------
Record the default NumPy run::

    python examples/environments/kuramoto_demo.py

Four parallel systems on the PyTorch backend, light theme::

    python examples/environments/kuramoto_demo.py --backend torch --n-agents 4 --theme light

Fixed distance-based coupling on two clusters, live window::

    python examples/environments/kuramoto_demo.py --mode constant --topology two_clusters --render-mode human
"""

from __future__ import annotations

import argparse

import numpy as np

import env_lib
from env_lib.utils import record_episode, rendering

TOPOLOGIES = ("fully_connected", "ring", "star", "random", "two_clusters")


def two_clusters(n: int) -> np.ndarray:
    """Adjacency matrix of two disconnected, fully connected clusters."""
    half = n // 2
    adjacency = np.zeros((n, n))
    adjacency[:half, :half] = 1.0
    adjacency[half:, half:] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def distance_based_coupling(n: int, base_strength: float = 2.0, decay: float = 0.5) -> np.ndarray:
    """Coupling ``base * exp(-decay * d_ij)`` with ``d_ij`` the ring distance of i and j."""
    index = np.arange(n)
    offset = np.abs(index[:, None] - index[None, :])
    distance = np.minimum(offset, n - offset)
    coupling = base_strength * np.exp(-decay * distance)
    np.fill_diagonal(coupling, 0.0)
    return coupling


class SyncController:
    """Coupling ramp (dynamic mode) plus proportional phase alignment.

    Parameters
    ----------
    env:
        The (possibly wrapped) Kuramoto environment.
    gain:
        Gain of the phase-alignment control input.
    ramp_steps:
        Number of steps over which the coupling strengths rise to their maximum.
    log_every:
        Print progress every ``log_every`` steps (0 disables printing).
    """

    def __init__(self, env, gain: float = 1.0, ramp_steps: int = 150, log_every: int = 25):
        self.core = env.unwrapped
        self.gain = gain
        self.ramp_steps = max(1, ramp_steps)
        self.log_every = log_every
        self.batched = hasattr(self.core, "get_batch_observations") and self.core.n_agents > 1
        self.step = 0

    def _action(self, observations: np.ndarray) -> np.ndarray:
        core, n = self.core, self.core.n_oscillators
        phases = observations[..., :n].astype(np.float64)
        mean = np.exp(1j * phases).mean(axis=-1, keepdims=True)
        low, high = core.control_input_range
        control = np.clip(self.gain * np.abs(mean) * np.sin(np.angle(mean) - phases), low, high)
        if core.coupling_mode != "dynamic":
            return control.astype(np.float32)
        level = min(1.0, self.step / self.ramp_steps)
        c_low, c_high = core.coupling_range
        coupling = np.full(
            control.shape[:-1] + (core.n_couplings,), c_low + level * (c_high - c_low)
        )
        return np.concatenate((control, coupling), axis=-1).astype(np.float32)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if self.batched:
            observation = self.core.get_batch_observations().cpu().numpy()
        action = self._action(observation)
        if self.log_every and self.step % self.log_every == 0:
            n = self.core.n_oscillators
            r = np.abs(np.exp(1j * observation[..., :n].astype(np.float64)).mean(axis=-1))
            coupling = (
                f" | coupling {action[..., n:].max():.2f}"
                if self.core.coupling_mode == "dynamic"
                else ""
            )
            print(
                f"step {self.step:4d} | t={self.step * self.core.dt:5.2f}s | "
                f"r={np.mean(r):.3f}{coupling}"
            )
        self.step += 1
        return action


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=300, help="maximum number of steps")
    parser.add_argument("--seed", type=int, default=0, help="reset seed")
    parser.add_argument("--save", default="renders/kuramoto.gif", help="output .gif or .mp4 file")
    parser.add_argument(
        "--render-mode", choices=("rgb_array", "human", "none"), default="rgb_array"
    )
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--n-oscillators", type=int, default=12)
    parser.add_argument("--n-agents", type=int, default=1, help="parallel systems (torch only)")
    parser.add_argument("--mode", choices=("dynamic", "constant"), default="dynamic")
    parser.add_argument("--topology", choices=TOPOLOGIES, default="fully_connected")
    parser.add_argument("--gain", type=float, default=0.8, help="phase-alignment gain")
    parser.add_argument("--ramp-steps", type=int, default=150, help="coupling ramp length")
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="divide the coupling sum by N (classical K/N model)",
    )
    parser.add_argument("--render-every", type=int, default=2, help="record every k-th step")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    return parser.parse_args(argv)


def make_env(args: argparse.Namespace):
    n = args.n_oscillators
    kwargs = dict(
        n_oscillators=n,
        max_steps=args.steps,
        coupling_mode=args.mode,
        normalize_coupling=args.normalize,
        render_mode=None if args.render_mode == "none" else args.render_mode,
    )
    if args.topology == "two_clusters":
        kwargs["adj_matrix"] = two_clusters(n)
    else:
        kwargs["topology"] = args.topology
    if args.mode == "constant":
        coupling = distance_based_coupling(n, base_strength=4.0 if args.normalize else 0.4)
        if args.topology == "two_clusters":
            coupling = coupling * kwargs["adj_matrix"]
        kwargs["constant_coupling_matrix"] = coupling
    if args.backend == "torch":
        return env_lib.KuramotoOscillatorEnvTorch(n_agents=args.n_agents, device="auto", **kwargs)
    return env_lib.KuramotoOscillatorEnv(**kwargs)


def main(argv: list | None = None) -> None:
    args = parse_args(argv)
    rendering.set_theme(args.theme)
    env = make_env(args)
    controller = SyncController(env, gain=args.gain, ramp_steps=args.ramp_steps)
    core = env.unwrapped
    print(
        f"{args.backend} backend | N={core.n_oscillators} | {core.coupling_mode} coupling | "
        f"{core.topology} topology | sync threshold {core.sync_threshold}"
    )

    if args.render_mode == "rgb_array":
        frames = record_episode(
            env,
            controller,
            args.save,
            max_steps=args.steps,
            seed=args.seed,
            render_every=args.render_every,
        )
        print(f"saved {len(frames)} frames to {args.save}")
    else:
        observation, _ = env.reset(seed=args.seed)
        for _ in range(args.steps):
            observation, _, terminated, truncated, _ = env.step(controller(observation))
            if terminated or truncated:
                break

    phases = np.asarray(core.phase_history[-1]).reshape(-1, core.n_oscillators)[0]
    r = float(np.abs(np.exp(1j * phases).mean()))
    status = "synchronised" if r > core.sync_threshold else "not synchronised"
    print(f"finished after {core.step_count} steps: r={r:.3f} ({status})")
    env.close()


if __name__ == "__main__":
    main()
