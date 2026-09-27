"""Configuration dataclasses shared by the algorithm families."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

__all__ = ["AlgorithmConfig", "OffPolicyConfig", "OnPolicyConfig"]


@dataclass
class AlgorithmConfig:
    """Options common to every algorithm.

    Attributes
    ----------
    hidden_sizes:
        Widths of the hidden layers of every network.
    activation:
        Hidden activation (``"tanh"``, ``"relu"``, ``"elu"``, ``"gelu"``).
    gamma:
        Discount factor.
    share_parameters:
        One set of network weights for all agents (standard for homogeneous
        agents; scales to many agents).
    agent_ids:
        Append a one-hot agent identifier to per-agent inputs when parameters
        are shared, so agents can still specialise.
    max_grad_norm:
        Gradient-norm clipping threshold.
    """

    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    gamma: float = 0.99
    share_parameters: bool = True
    agent_ids: bool = True
    max_grad_norm: float = 10.0

    def __post_init__(self) -> None:
        self.hidden_sizes = tuple(int(h) for h in self.hidden_sizes)
        if not self.hidden_sizes or min(self.hidden_sizes) < 1:
            raise ValueError(f"hidden_sizes must be positive, got {self.hidden_sizes}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")

    @classmethod
    def field_names(cls) -> list[str]:
        """Names of all configuration fields."""
        return [f.name for f in fields(cls)]

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary of the configuration."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class OnPolicyConfig(AlgorithmConfig):
    """Options of on-policy (actor-critic) methods.

    Attributes
    ----------
    rollout_length:
        Vector steps collected per update (``rollout_length * num_envs``
        transitions per copy set).
    lr:
        Adam learning rate.
    gae_lambda:
        GAE parameter.
    normalize_observations:
        Standardise observations with running statistics.
    normalize_rewards:
        Scale rewards by the running standard deviation of the return.
    """

    max_grad_norm: float = 0.5
    rollout_length: int = 64
    lr: float = 3e-4
    gae_lambda: float = 0.95
    normalize_observations: bool = True
    normalize_rewards: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rollout_length < 1:
            raise ValueError(f"rollout_length must be positive, got {self.rollout_length}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")


@dataclass
class OffPolicyConfig(AlgorithmConfig):
    """Options of off-policy (replay-based) methods.

    Attributes
    ----------
    buffer_size:
        Replay capacity in transitions (one per copy and step).
    batch_size:
        Transitions per gradient step.
    warmup_steps:
        Environment steps (summed over copies) with uniformly random actions
        before learning starts.
    update_every:
        Vector steps between rounds of gradient steps.
    gradient_steps:
        Gradient steps per round.
    tau:
        Polyak coefficient of the target networks.
    reward_scale:
        Constant factor applied to rewards before learning.
    """

    activation: str = "relu"
    buffer_size: int = 200_000
    batch_size: int = 256
    warmup_steps: int = 2_000
    update_every: int = 1
    gradient_steps: int = 1
    tau: float = 0.01
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("buffer_size", "batch_size", "update_every", "gradient_steps"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
