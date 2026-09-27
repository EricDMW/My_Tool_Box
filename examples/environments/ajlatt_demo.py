"""AJLATT demo: cooperative target tracking with the packaged baseline.

The baseline (``env_lib.baseline_policy``, implemented by
``env_lib.baselines.ajlatt_encircle``) drives each robot towards a slot on a
circle around its current belief of the target position; the slots are spread
evenly, so the team views the target from several directions. It is computed
from the observation alone. The script compares it with random actions, prints
per-episode metrics and can record a GIF of the baseline.

Run::

    python examples/environments/ajlatt_demo.py --episodes 3
    python examples/environments/ajlatt_demo.py --map obstacles05 --save renders/ajlatt.gif
    python examples/environments/ajlatt_demo.py --render-mode human
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env_lib import AJLATTEnv, baseline_policy
from env_lib.utils import record_episode, set_theme


def run_episode(env: AJLATTEnv, policy: str, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    total = np.zeros(env.num_robots)
    collisions = 0
    controller = baseline_policy(env)
    while True:
        if policy == "heuristic":
            action = controller(obs)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        collisions += int(info["collisions"].sum())
        if truncated:
            break
    stats = env.episode_statistics()
    return {
        "return": float(total.sum()),
        "target_error": float(stats["target_error"][-20:].mean()),
        "target_cov": float(stats["target_cov_trace"][-20:].mean()),
        "collisions": collisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--map", default="obstacles04", help="map name (see ajlatt-plot-map --list)"
    )
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120, help="episode length")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save", default=None, help="GIF path for a heuristic episode, e.g. renders/ajlatt.gif"
    )
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default=None)
    parser.add_argument("--theme", choices=["dark", "light"], default="dark")
    args = parser.parse_args()
    set_theme(args.theme)

    env = AJLATTEnv(map_name=args.map, num_robots=args.robots, max_episode_steps=args.steps)
    print(f"AJLATT on {args.map}: {args.robots} robots, {args.steps} steps per episode")
    print(
        f"{'policy':>10} {'episode':>8} {'return':>10} {'target err [m]':>15} {'target tr(P)':>13} {'collisions':>11}"
    )
    for policy in ("random", "heuristic"):
        for episode in range(args.episodes):
            m = run_episode(env, policy, args.seed + episode)
            print(
                f"{policy:>10} {episode:>8d} {m['return']:>10.1f} {m['target_error']:>15.3f} "
                f"{m['target_cov']:>13.3f} {m['collisions']:>11d}"
            )
    env.close()

    if args.save or args.render_mode == "rgb_array":
        path = Path(args.save or "renders/ajlatt.gif")
        video_env = AJLATTEnv(
            map_name=args.map,
            num_robots=args.robots,
            max_episode_steps=args.steps,
            render_mode="rgb_array",
        )
        frames = record_episode(video_env, baseline_policy(video_env), path, seed=args.seed)
        print(f"saved {len(frames)} frames to {path}")
        video_env.close()
    elif args.render_mode == "human":
        human_env = AJLATTEnv(
            map_name=args.map,
            num_robots=args.robots,
            max_episode_steps=args.steps,
            render_mode="human",
        )
        obs, _ = human_env.reset(seed=args.seed)
        controller = baseline_policy(human_env)
        for _ in range(args.steps):
            obs, *_ = human_env.step(controller(obs))
        human_env.close()


if __name__ == "__main__":
    main()
