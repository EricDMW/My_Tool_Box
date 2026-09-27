"""Tuned settings of IQL, VDN and QMIX for the env_lib environments."""

from __future__ import annotations

import copy
from typing import Any

__all__ = ["PRESETS"]

#: LineMsg-v0 (10 agents): every method learns "always relay" (return 95, the
#: optimum; random actions about 43) in about 10-20 s.
_LINEMSG: dict[str, Any] = {
    "num_envs": 16,
    "total_steps": 25_000,
    "config": {"batch_size": 128, "lr": 1e-3, "epsilon_decay_steps": 12_500},
    "env_kwargs": {},
}

#: WirelessComm-v1 (4x4 grid, 16 agents): packets are delivered in the step
#: they are sent, so a short horizon (gamma = 0.8) suffices and learns faster.
#: The collision-free schedule of env_lib.baseline_policy returns about 340,
#: random actions about 140. VDN and QMIX return 320-345 depending on the seed:
#: they learn either a collision-free "owners only" schedule (about 319: every
#: access point serves one fixed neighbouring agent) or, like the baseline,
#: also let the border agents borrow idle access points.
_WIRELESS_CONFIG: dict[str, Any] = {"batch_size": 128, "lr": 1e-3, "gamma": 0.8}

#: Tuned demonstration settings ``PRESETS[algorithm][env_id]`` for ``algorithm``
#: in ``"iql"``, ``"vdn"``, ``"qmix"``. A preset is a dictionary with the keys
#: ``"total_steps"`` (environment steps summed over copies), ``"num_envs"``
#: (batched copies), ``"config"`` (:class:`QLearningConfig` overrides) and
#: ``"env_kwargs"`` (environment arguments); see :mod:`marl_algorithms.presets`.
#: Every preset trains in at most about two minutes on one CPU core.
#:
#: IQL has no WirelessComm preset: with its per-agent rewards a collision costs
#: the transmitting agent nothing, so independent learners keep transmitting
#: and collide (hardly better than random actions), while VDN and QMIX, trained
#: on the team reward, learn to leave contested access points to one agent.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "iql": {
        "LineMsg-v0": copy.deepcopy(_LINEMSG),
    },
    "vdn": {
        "LineMsg-v0": copy.deepcopy(_LINEMSG),
        "WirelessComm-v1": {
            "num_envs": 16,
            "total_steps": 120_000,
            "config": {**_WIRELESS_CONFIG, "epsilon_decay_steps": 30_000},
            "env_kwargs": {},
        },
    },
    "qmix": {
        "LineMsg-v0": copy.deepcopy(_LINEMSG),
        "WirelessComm-v1": {
            "num_envs": 16,
            "total_steps": 100_000,
            "config": {**_WIRELESS_CONFIG, "epsilon_decay_steps": 45_000},
            "env_kwargs": {},
        },
    },
}
