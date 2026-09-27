"""Quick start: create every environment, run a few random steps, render a frame.

Run::

    python examples/quickstart.py
    python examples/quickstart.py --render renders/quickstart
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

import env_lib
from env_lib.utils import save_animation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=20, help="random steps per environment")
    parser.add_argument("--render", metavar="DIR", help="save one PNG frame per environment to DIR")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print(f"env_lib {env_lib.__version__}: {len(env_lib.list_envs())} registered environments")
    for env_id in env_lib.list_envs():
        try:
            env = env_lib.make(env_id, render_mode="rgb_array" if args.render else None)
        except ImportError as exc:  # optional dependency (torch, pygame, pymunk) missing
            print(f"  {env_id:<46} skipped: requires {exc.name}")
            continue
        obs, _ = env.reset(seed=0)
        total = 0.0
        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += float(np.sum(reward))
            if np.any(terminated) or truncated:
                obs, _ = env.reset()
        print(
            f"  {env_id:<46} obs {str(np.shape(obs)):<12} return over {args.steps} steps: {total:10.2f}"
        )
        if args.render:
            out = Path(args.render) / f"{env_id}.gif"
            save_animation([env.render()], out, fps=1)
        env.close()


if __name__ == "__main__":
    main()
