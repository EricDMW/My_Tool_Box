"""marl_algorithms: classical multi-agent reinforcement learning algorithms.

Reference implementations of seven widely used methods, written against one
small core and trained directly on the ``env_lib`` environments:

==========  ========================  ==========================
Algorithm   Family                    Actions
==========  ========================  ==========================
IPPO        on-policy                 continuous and discrete
MAPPO       on-policy                 continuous and discrete
MADDPG      off-policy actor-critic   continuous
MATD3       off-policy actor-critic   continuous
IQL         value-based               discrete
VDN         value-based               discrete
QMIX        value-based               discrete
==========  ========================  ==========================

Layout::

    core/          shared building blocks: agent spec, experience collection,
                   buffers, normalisation, networks, base classes
    algorithms/    one module per algorithm, grouped by family:
                   ppo/ (IPPO, MAPPO), ddpg/ (MADDPG, MATD3),
                   q_learning/ (IQL, VDN, QMIX)
    presets/       tuned settings for the env_lib environments
    registry.py    algorithm names and train()
    baselines.py   compare(): the algorithms and classical controllers as
                   baselines for your own method
    __main__.py    the marl-train command line

Quick start::

    import env_lib
    from marl_algorithms import compare, train

    algo, log = train("mappo", "PowerGrid-v0", total_steps=200_000, num_envs=32)
    print(log.summary())
    print(algo.evaluate(env_lib.make_vec("PowerGrid-v0", 16), n_episodes=32))

    # Baselines: every preset algorithm, random actions, the classical
    # controller and your policy, on the same evaluation episodes
    print(compare("PowerGrid-v0", seeds=(0, 1, 2), policies={"mine": my_policy}))

The package requires PyTorch (``pip install "my-tool-box[torch]"``).
"""

from __future__ import annotations

import importlib
from typing import Any

from marl_algorithms._version import __version__

_LAZY: dict[str, str] = {
    # Core
    "Algorithm": "marl_algorithms.core.base",
    "OnPolicyAlgorithm": "marl_algorithms.core.base",
    "OffPolicyAlgorithm": "marl_algorithms.core.base",
    "TrainingLog": "marl_algorithms.core.base",
    "MultiAgentSpec": "marl_algorithms.core.spec",
    "VectorRunner": "marl_algorithms.core.runner",
    "Transition": "marl_algorithms.core.runner",
    "make_vector_env": "marl_algorithms.core.runner",
    "torch_threads": "marl_algorithms.core.base",
    # Registry
    "AlgorithmInfo": "marl_algorithms.registry",
    "get_algorithm": "marl_algorithms.registry",
    "list_algorithms": "marl_algorithms.registry",
    "make_algorithm": "marl_algorithms.registry",
    "train": "marl_algorithms.registry",
    # Presets
    "get_preset": "marl_algorithms.presets",
    "list_presets": "marl_algorithms.presets",
    "train_preset": "marl_algorithms.presets",
    # Baselines
    "Comparison": "marl_algorithms.baselines",
    "ComparisonRow": "marl_algorithms.baselines",
    "compare": "marl_algorithms.baselines",
    "per_copy": "marl_algorithms.baselines",
    # Algorithms
    "IPPO": "marl_algorithms.algorithms.ppo",
    "MAPPO": "marl_algorithms.algorithms.ppo",
    "PPOConfig": "marl_algorithms.algorithms.ppo",
    "MADDPG": "marl_algorithms.algorithms.ddpg",
    "MATD3": "marl_algorithms.algorithms.ddpg",
    "DDPGConfig": "marl_algorithms.algorithms.ddpg",
    "IQL": "marl_algorithms.algorithms.q_learning",
    "VDN": "marl_algorithms.algorithms.q_learning",
    "QMIX": "marl_algorithms.algorithms.q_learning",
    "QLearningConfig": "marl_algorithms.algorithms.q_learning",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        try:
            module = importlib.import_module(_LAZY[name])
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise ImportError(
                f"marl_algorithms.{name} requires PyTorch; install it with "
                'pip install "my-tool-box[torch]"'
            ) from exc
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
