"""Pistonball demo: a hand-written heuristic versus random actions.

The pistons must roll the ball to the left wall. The heuristic uses only the
joint observation: it reads the ball position from the pistons that observe
the ball (the ``kappa``-hop neighbourhood) and shapes the pistons into a ramp
that falls towards the left (pistons right of the ball up, pistons left of it
down). Both policies are evaluated on the same seeds and the script prints the
mean team return, the success rate (ball reached the left wall) and the mean
episode length.

Examples
--------
Compare the policies (headless)::

    python examples/environments/pistonball_demo.py --episodes 10

Discrete actions, wider observation range and a movement penalty::

    python examples/environments/pistonball_demo.py --discrete --kappa 2 --movement-penalty -0.05

Record one heuristic episode as a GIF (works headless)::

    python examples/environments/pistonball_demo.py --save renders/pistonball.gif --theme light

Watch the heuristic in a window, or play yourself (W/S move the selected
piston, A/D change the selection, Backspace resets, Esc quits)::

    python examples/environments/pistonball_demo.py --render-mode human --episodes 2
    python examples/environments/pistonball_demo.py --manual --n-pistons 8
"""

from __future__ import annotations

import argparse
import os
from typing import Callable

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from env_lib.pistonball_env import ManualPolicy, PistonballEnv
from env_lib.utils import record_episode, set_theme

Policy = Callable[[np.ndarray], np.ndarray]


def make_heuristic_policy(n_pistons: int, continuous: bool) -> Policy:
    """Ramp heuristic computed from the joint observation ``(n_pistons, 7)``.

    Column 1 is the piston x position ``(5 + 40 i) / (40 n)`` and column 2 the
    ball x position ``(x - 80) / (40 n)`` (zero for pistons that do not see the
    ball), so ``(obs[:, 1] - ball_x) * n`` is the signed distance from piston
    ``i`` to the ball in piston widths. The target height rises from fully down
    (half a piston left of the ball) to fully up (half a piston right of it).
    """

    def policy(obs: np.ndarray) -> np.ndarray:
        observers = np.any(obs[:, 2:] != 0.0, axis=1)
        if not observers.any():
            action = np.zeros(n_pistons)
        else:
            ball_x = obs[observers, 2][0]
            offset = (obs[:, 1] - ball_x) * n_pistons + 15.0 / 40.0  # lane centre - ball
            raised = np.clip(0.5 + offset, 0.0, 1.0)  # 1 = fully up
            target = 1.0 - 2.0 * raised  # in units of observation column 0
            action = np.clip(8.0 * (obs[:, 0] - target), -1.0, 1.0)  # +1 moves up
        if continuous:
            return action.astype(np.float32)
        return np.rint(action).astype(np.int64) + 1  # {0: down, 1: stay, 2: up}

    return policy


def make_random_policy(env: PistonballEnv, seed: int) -> Policy:
    env.action_space.seed(seed)
    return lambda _obs: env.action_space.sample()


def evaluate(env: PistonballEnv, policy: Policy, episodes: int, seed: int) -> dict[str, float]:
    """Run ``episodes`` episodes and summarise returns, successes and lengths."""
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total, steps = 0.0, 0
        while True:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += reward
            steps += 1
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(steps)
        successes += int(terminated)
    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "success": successes / episodes,
        "length": float(np.mean(lengths)),
    }


def play_manually(env_kwargs: dict) -> None:
    env = PistonballEnv(render_mode="human", **env_kwargs)
    policy = ManualPolicy(env)
    obs, _ = env.reset()
    total = 0.0
    selected = policy.selected_piston
    print("W/S: move piston, A/D: select piston, Backspace: reset, Esc: quit")
    while policy.running:
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        if policy.selected_piston != selected:
            selected = policy.selected_piston
            print(f"selected piston {selected}")
        if terminated or truncated:
            print(
                f"episode finished: return {total:.2f} ({'goal' if terminated else 'time limit'})"
            )
            obs, _ = env.reset()
            total = 0.0
        if env.renderer is not None and not env.renderer.is_open:
            break
    env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-pistons", type=int, default=20, help="number of pistons (agents)")
    parser.add_argument("--episodes", type=int, default=5, help="episodes per policy")
    parser.add_argument("--steps", type=int, default=125, help="episode length (max_cycles)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first episode")
    parser.add_argument("--kappa", type=int, default=1, help="observation radius in pistons")
    parser.add_argument("--discrete", action="store_true", help="use {0, 1, 2} actions")
    parser.add_argument("--movement-penalty", type=float, default=0.0, help="per 4 px moved")
    parser.add_argument(
        "--render-mode", choices=["none", "human"], default="none", help="show the heuristic"
    )
    parser.add_argument("--save", metavar="PATH", help="record a heuristic episode (.gif/.mp4)")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark")
    parser.add_argument("--manual", action="store_true", help="keyboard control in a window")
    args = parser.parse_args(argv)

    set_theme(args.theme)
    env_kwargs = dict(
        n_pistons=args.n_pistons,
        max_cycles=args.steps,
        kappa=args.kappa,
        continuous=not args.discrete,
        movement_penalty=args.movement_penalty,
    )
    if args.manual:
        play_manually(env_kwargs)
        return

    heuristic = make_heuristic_policy(args.n_pistons, continuous=not args.discrete)
    mode = "discrete" if args.discrete else "continuous"
    print(
        f"Pistonball: {args.n_pistons} pistons, kappa={args.kappa}, {mode} actions, "
        f"{args.episodes} episodes of at most {args.steps} steps"
    )
    print(f"{'policy':<10} {'return':>18} {'success':>8} {'length':>7}")
    shown = (
        PistonballEnv(render_mode="human", **env_kwargs) if args.render_mode == "human" else None
    )
    headless = PistonballEnv(**env_kwargs)
    for name in ("heuristic", "random"):
        env = shown if (shown is not None and name == "heuristic") else headless
        policy = heuristic if name == "heuristic" else make_random_policy(env, args.seed)
        stats = evaluate(env, policy, args.episodes, args.seed)
        print(
            f"{name:<10} {stats['mean']:9.2f} +- {stats['std']:6.2f} "
            f"{100 * stats['success']:7.0f}% {stats['length']:7.1f}"
        )
    headless.close()
    if shown is not None:
        shown.close()

    if args.save:
        recorder = PistonballEnv(render_mode="rgb_array", **env_kwargs)
        frames = record_episode(recorder, heuristic, args.save, seed=args.seed)
        recorder.close()
        print(f"saved {len(frames)} frames to {args.save}")


if __name__ == "__main__":
    main()
