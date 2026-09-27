"""Networked consensus (rendezvous) and formation control.

Registered as ``Consensus-v0`` and ``Formation-v0``::

    import env_lib

    env = env_lib.make("Formation-v0", formation_shape="wedge", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    action = env.unwrapped.laplacian_policy()

The renderer (:mod:`env_lib.consensus_env.rendering`) is imported lazily on the
first ``render()`` call.
"""

from __future__ import annotations

from env_lib.consensus_env.consensus_env import (
    DYNAMICS,
    FORMATION_SHAPES,
    TASKS,
    TOPOLOGIES,
    ConsensusEnv,
    algebraic_connectivity,
    is_connected,
    make_formation,
    make_topology,
    proximity_adjacency,
)

__all__ = [
    "DYNAMICS",
    "FORMATION_SHAPES",
    "TASKS",
    "TOPOLOGIES",
    "ConsensusEnv",
    "algebraic_connectivity",
    "is_connected",
    "make_formation",
    "make_topology",
    "proximity_adjacency",
]
