"""Line message passing: random actions versus simple relay heuristics.

In the ``LineMsg`` environment a line of agents relays a message from the
source (right end) to the sink (left end, agent 0). A middle agent that acts
(action 1) receives the message from its right neighbour, or obtains one with
probability 0.8 when the neighbour is empty; an agent that does not act loses
the message. The sink always copies agent 1 and earns reward 1.0 while it
holds the message, every other agent earns 0.1.

This script compares three decentralised policies over a batch of episodes:

``random``
    Every agent acts with probability 1/2.
``duty``
    Duty-cycled relays: every agent is awake (acts) with probability
    ``--duty``, e.g. to save energy. A sleeping agent breaks the chain, and
    the message front has to travel back to the sink.
``relay``
    Every agent always acts. Acting has no cost in this environment, so this
    is the best policy (the line never loses the message).

It prints the mean episode return of each policy and can show or record one
episode of the chosen policy.

Run it with::

    python examples/linemsg_demo.py
    python examples/linemsg_demo.py --num-agents 30 --action-space-type multibinary
    python examples/linemsg_demo.py --duty 0.8 --save renders/linemsg.gif
    python examples/linemsg_demo.py --render-mode human
"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from env_lib.linemsg_env import LineMsgEnv
from env_lib.utils import record_episode, set_theme

Policy = Callable[[np.ndarray], np.ndarray]


def make_policy(name: str, env: LineMsgEnv, seed: int, duty: float) -> Policy:
    """Return a policy mapping the ``(num_agents, window)`` observation to binary actions."""
    rng = np.random.default_rng(seed)
    awake = {"random": 0.5, "duty": duty, "relay": 1.0}[name]
    return lambda obs: (rng.random(env.num_agents) < awake).astype(np.int64)


def evaluate(env: LineMsgEnv, policy: Policy, episodes: int, seed: int) -> dict[str, float]:
    returns, sink = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total, sink_steps, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            sink_steps += int(info["agent_rewards"][0] > 0)
            done = terminated or truncated
        returns.append(total)
        sink.append(sink_steps / env.max_iter)
    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "sink": float(np.mean(sink)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--n-obs-neighbors", type=int, default=1)
    parser.add_argument(
        "--action-space-type", choices=("discrete", "multibinary"), default="discrete"
    )
    parser.add_argument("--steps", type=int, default=50, help="episode length (max_iter)")
    parser.add_argument("--episodes", type=int, default=20, help="evaluation episodes per policy")
    parser.add_argument("--duty", type=float, default=0.9, help="awake probability of 'duty'")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=("random", "duty", "relay"),
        default="duty",
        help="policy shown with --render-mode human or recorded with --save",
    )
    parser.add_argument("--render-mode", choices=("none", "human", "ansi"), default="none")
    parser.add_argument("--save", default=None, help="record one episode, e.g. renders/linemsg.gif")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_theme(args.theme)
    config = dict(
        num_agents=args.num_agents,
        n_obs_neighbors=args.n_obs_neighbors,
        max_iter=args.steps,
        action_space_type=args.action_space_type,
    )

    env = LineMsgEnv(**config)
    print(
        f"LineMsg with {args.num_agents} agents, {args.steps} steps, {args.episodes} episodes, "
        f"duty={args.duty}"
    )
    print(f"{'policy':<8} {'return':>16} {'sink holds msg':>15}")
    for name in ("random", "duty", "relay"):
        stats = evaluate(
            env, make_policy(name, env, args.seed, args.duty), args.episodes, args.seed
        )
        print(f"{name:<8} {stats['mean']:9.2f} +- {stats['std']:4.2f} {stats['sink']:14.0%}")
    env.close()

    if args.render_mode != "none":
        env = LineMsgEnv(render_mode=args.render_mode, **config)
        policy = make_policy(args.policy, env, args.seed, args.duty)
        obs, _ = env.reset(seed=args.seed)
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            done = terminated or truncated
        if args.render_mode == "ansi":
            print(env.render(), end="")
        env.close()

    if args.save:
        env = LineMsgEnv(render_mode="rgb_array", **config)
        policy = make_policy(args.policy, env, args.seed, args.duty)
        frames = record_episode(env, policy, args.save, seed=args.seed)
        env.close()
        print(f"saved {len(frames)} frames of the '{args.policy}' policy to {args.save}")


if __name__ == "__main__":
    main()
