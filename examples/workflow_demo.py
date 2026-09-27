"""End-to-end research workflow with env_lib: discover, vectorise, evaluate, adapt, record.

The script walks through the convenience layer in six short steps:

1. list the environments with continuous actions (``env_lib.catalog``);
2. create a natively batched vector environment (``env_lib.make_vec``);
3. evaluate random actions and the classical baseline controller on all
   copies in parallel (``env_lib.evaluate``, ``env_lib.baseline_policy``);
4. flatten the joint spaces for single-agent libraries such as
   Stable-Baselines3 (``FlattenJointSpaces``);
5. expose the team through the PettingZoo Parallel API (``to_parallel``);
6. record a GIF of the baseline controller (``record_episode``).

Run::

    python examples/workflow_demo.py
    python examples/workflow_demo.py --env Consensus-v0 --num-envs 128 --episodes 128
    python examples/workflow_demo.py --save renders/workflow.gif --steps 120
    python examples/workflow_demo.py --no-save

The same steps are available on the command line: ``env-lib list --continuous``,
``env-lib evaluate Formation-v0 --num-envs 32`` and
``env-lib run Formation-v0 --gif renders/formation.gif``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import env_lib
from env_lib.utils import record_episode
from env_lib.wrappers import FlattenJointSpaces, to_parallel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", default="Formation-v0", help="registered environment id")
    parser.add_argument("--num-envs", type=int, default=32, help="parallel copies")
    parser.add_argument("--episodes", type=int, default=64, help="evaluation episodes per policy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=150, help="maximum GIF length in steps")
    parser.add_argument("--save", default="renders/workflow.gif", help="GIF of the baseline")
    parser.add_argument("--no-save", action="store_true", help="skip the GIF")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 1. Discover: every environment whose actions can be continuous.
    print("1. Environments with continuous actions\n")
    print(env_lib.catalog(action_type="continuous", available_only=True, inspect=False))

    # 2. Vectorise: make_vec picks the native batched implementation when one exists.
    envs = env_lib.make_vec(args.env, num_envs=args.num_envs)
    kind = type(envs.unwrapped).__name__
    print(
        f"\n2. {args.env} x {args.num_envs} copies: {kind}, observations {envs.observation_space.shape}"
    )

    # 3. Evaluate random actions and the baseline on all copies in parallel.
    print("\n3. Parallel evaluation\n")
    policy = env_lib.baseline_policy(envs)
    print(f"baseline controller: {policy.description}\n")
    for candidate in (None, policy):
        started = time.perf_counter()
        result = env_lib.evaluate(envs, candidate, n_episodes=args.episodes, seed=args.seed)
        print(result)
        print(f"  ({time.perf_counter() - started:.2f} s)\n")
    envs.close()

    # 4. Single-agent view: a flat observation vector and a flat action.
    flat = FlattenJointSpaces(env_lib.make(args.env))
    observation, _ = flat.reset(seed=args.seed)
    flat_policy = env_lib.baseline_policy(flat)  # the baseline adapts to the flat spaces
    _, reward, *_ = flat.step(flat_policy(observation))
    print(
        f"4. Flattened for single-agent libraries: observation {flat.observation_space.shape}, "
        f"action {flat.action_space.shape}, first reward {reward:.3f}"
    )
    print('   e.g. stable_baselines3.PPO("MlpPolicy", flat).learn(100_000)')
    flat.close()

    # 5. Multi-agent view: the PettingZoo Parallel API (dictionaries keyed by agent).
    par_env = to_parallel(args.env)
    observations, _ = par_env.reset(seed=args.seed)
    par_policy = env_lib.baseline_policy(par_env)  # dict of observations -> dict of actions
    _, rewards, terminations, truncations, _ = par_env.step(par_policy(observations))
    mean_reward = np.mean(list(rewards.values()))
    print(
        f"\n5. PettingZoo Parallel API: {len(par_env.possible_agents)} agents "
        f"({par_env.possible_agents[0]}, ...), per-agent observation "
        f"{par_env.observation_space(par_env.possible_agents[0]).shape}, "
        f"mean agent reward {mean_reward:.3f}"
    )
    par_env.close()

    # 6. Record the baseline.
    if not args.no_save:
        env = env_lib.make(args.env, render_mode="rgb_array")
        frames = record_episode(
            env, env_lib.baseline_policy(env), args.save, seed=args.seed, max_steps=args.steps
        )
        env.close()
        print(f"\n6. Saved {len(frames)} frames of the baseline to {args.save}")


if __name__ == "__main__":
    main()
