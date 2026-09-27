"""Wireless multiple access: random access versus simple coordination heuristics.

The ``WirelessComm`` environment places agents on a grid; every agent holds a
queue of packets with deadlines and may send its earliest packet to one of the
(up to) four access points at the corners of its grid cell. Two agents that
use the same access point in the same step collide and both fail.

This script compares three decentralised policies over a batch of episodes:

``random``
    Uniformly random actions (the usual baseline).
``aloha``
    p-persistent random access: an agent holding a packet transmits with
    probability ``--tau`` to a uniformly chosen access point that exists.
``schedule``
    A collision-free rotating schedule: at step ``t`` every agent uses the
    same direction ``(4, 1, 2, 3)[t % 4]``. A common offset maps agents to
    access points one-to-one, so no two agents ever share an access point.

It prints the mean episode return of each policy and can show or record one
episode of the chosen policy.

Run it with::

    python examples/wireless_comm_demo.py
    python examples/wireless_comm_demo.py --grid 8 8 --tau 0.5 --episodes 50
    python examples/wireless_comm_demo.py --policy aloha --save renders/wireless_comm.gif
    python examples/wireless_comm_demo.py --render-mode human
"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from env_lib.utils import record_episode, set_theme
from env_lib.wireless_comm_env import WirelessCommEnv
from env_lib.wireless_comm_env.wireless_comm_env import OUTCOME_COLLISION

Policy = Callable[[np.ndarray], np.ndarray]


def own_queue_columns(env: WirelessCommEnv) -> np.ndarray:
    """Observation columns that hold an agent's own queue (one per deadline slot)."""
    n, w = env.n_obs_nghbr, env.window
    return np.array([d * w * w + n * w + n for d in range(env.ddl)])


def valid_actions(env: WirelessCommEnv) -> list[np.ndarray]:
    """For every agent, the transmit actions whose access point exists."""
    table = []
    for k in range(env.n_agents):
        i, j = divmod(k, env.grid_y)
        table.append(
            np.array([a for a in range(1, 5) if env.access_point_mapping(i, j, a)[0] is not None])
        )
    return table


def make_random_policy(env: WirelessCommEnv, rng: np.random.Generator) -> Policy:
    def policy(obs: np.ndarray) -> np.ndarray:
        return rng.integers(5, size=env.n_agents)

    return policy


def make_aloha_policy(env: WirelessCommEnv, rng: np.random.Generator, tau: float) -> Policy:
    columns = own_queue_columns(env)
    choices = valid_actions(env)

    def policy(obs: np.ndarray) -> np.ndarray:
        has_packet = obs[:, columns].max(axis=1) > 0.5
        transmit = has_packet & (rng.random(env.n_agents) < tau)
        actions = np.zeros(env.n_agents, dtype=np.int64)
        for k in np.flatnonzero(transmit):
            actions[k] = rng.choice(choices[k])
        return actions

    return policy


def make_schedule_policy(env: WirelessCommEnv) -> Policy:
    columns = own_queue_columns(env)
    directions = (4, 1, 2, 3)
    # reachable[d, k]: agent k has an access point in direction directions[d].
    reachable = np.array(
        [[a in choices for choices in valid_actions(env)] for a in directions], dtype=bool
    )
    step = {"t": 0}

    def policy(obs: np.ndarray) -> np.ndarray:
        # A new episode starts from the initial state; restart the rotation.
        if env.num_moves == 0:
            step["t"] = 0
        phase = step["t"] % len(directions)
        step["t"] += 1
        has_packet = obs[:, columns].max(axis=1) > 0.5
        return np.where(has_packet & reachable[phase], directions[phase], 0)

    return policy


def make_policy(name: str, env: WirelessCommEnv, seed: int, tau: float) -> Policy:
    rng = np.random.default_rng(seed)
    if name == "random":
        return make_random_policy(env, rng)
    if name == "aloha":
        return make_aloha_policy(env, rng, tau)
    return make_schedule_policy(env)


def evaluate(env: WirelessCommEnv, policy: Policy, episodes: int, seed: int) -> dict[str, float]:
    returns, collisions = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total, n_coll, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            n_coll += int(np.sum(info["outcomes"] == OUTCOME_COLLISION))
            done = terminated or truncated
        returns.append(total)
        collisions.append(n_coll)
    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "collisions": float(np.mean(collisions)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", type=int, nargs=2, default=(6, 6), metavar=("X", "Y"))
    parser.add_argument("--ddl", type=int, default=2, help="deadline horizon (queue slots)")
    parser.add_argument("--p", type=float, default=0.8, help="packet arrival probability")
    parser.add_argument("--q", type=float, default=0.8, help="transmission success probability")
    parser.add_argument("--tau", type=float, default=0.6, help="ALOHA transmit probability")
    parser.add_argument("--steps", type=int, default=50, help="episode length (max_iter)")
    parser.add_argument("--episodes", type=int, default=20, help="evaluation episodes per policy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=("random", "aloha", "schedule"),
        default="schedule",
        help="policy shown with --render-mode human or recorded with --save",
    )
    parser.add_argument("--render-mode", choices=("none", "human"), default="none")
    parser.add_argument(
        "--save", default=None, help="record one episode, e.g. renders/wireless_comm.gif"
    )
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_theme(args.theme)
    config = dict(
        grid_x=args.grid[0],
        grid_y=args.grid[1],
        ddl=args.ddl,
        packet_arrival_probability=args.p,
        success_transmission_probability=args.q,
        max_iter=args.steps,
    )

    env = WirelessCommEnv(**config)
    print(
        f"WirelessComm {args.grid[0]}x{args.grid[1]}, ddl={args.ddl}, p={args.p}, q={args.q}, "
        f"{args.steps} steps, {args.episodes} episodes"
    )
    print(f"{'policy':<10} {'return':>16} {'collisions/ep':>14}")
    for name in ("random", "aloha", "schedule"):
        stats = evaluate(env, make_policy(name, env, args.seed, args.tau), args.episodes, args.seed)
        print(f"{name:<10} {stats['mean']:9.1f} +- {stats['std']:4.1f} {stats['collisions']:14.1f}")
    env.close()

    if args.render_mode == "human":
        env = WirelessCommEnv(render_mode="human", **config)
        policy = make_policy(args.policy, env, args.seed, args.tau)
        obs, _ = env.reset(seed=args.seed)
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            done = terminated or truncated
        env.close()

    if args.save:
        env = WirelessCommEnv(render_mode="rgb_array", **config)
        frames = record_episode(
            env, make_policy(args.policy, env, args.seed, args.tau), args.save, seed=args.seed
        )
        env.close()
        print(f"saved {len(frames)} frames of the '{args.policy}' policy to {args.save}")


if __name__ == "__main__":
    main()
