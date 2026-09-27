"""Neural-network building blocks shared by the algorithms.

All modules take per-agent inputs with arbitrary leading dimensions
``(..., features)``. Weights use orthogonal initialisation, the standard
choice for policy-gradient and actor-critic methods.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "CategoricalPolicy",
    "DeterministicPolicy",
    "GaussianPolicy",
    "PerAgent",
    "QMixer",
    "agent_one_hot",
    "mlp",
    "soft_update",
]

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}


def mlp(
    in_dim: int,
    out_dim: int,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: str = "tanh",
    *,
    output_gain: float = 1.0,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Multi-layer perceptron with orthogonal initialisation.

    Parameters
    ----------
    in_dim, out_dim:
        Input and output sizes.
    hidden_sizes:
        Widths of the hidden layers.
    activation:
        ``"tanh"``, ``"relu"``, ``"elu"`` or ``"gelu"``.
    output_gain:
        Orthogonal gain of the output layer (small values such as 0.01 give
        near-uniform initial policies).
    layer_norm:
        Apply layer normalisation after every hidden linear layer.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}")
    layers: list[nn.Module] = []
    size = int(in_dim)
    gain = nn.init.calculate_gain("tanh" if activation == "tanh" else "relu")
    for width in hidden_sizes:
        linear = nn.Linear(size, int(width))
        nn.init.orthogonal_(linear.weight, gain)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if layer_norm:
            layers.append(nn.LayerNorm(int(width)))
        layers.append(_ACTIVATIONS[activation]())
        size = int(width)
    head = nn.Linear(size, int(out_dim))
    nn.init.orthogonal_(head.weight, output_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


def agent_one_hot(
    batch_shape: tuple[int, ...], n_agents: int, device: torch.device | str
) -> torch.Tensor:
    """One-hot agent identifiers of shape ``(*batch_shape, n_agents, n_agents)``."""
    eye = torch.eye(n_agents, device=device)
    return eye.expand(*batch_shape, n_agents, n_agents)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging ``target <- (1 - tau) target + tau source``."""
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(1.0 - tau).add_(s, alpha=tau)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy with a state-independent log standard deviation.

    Actions are sampled unbounded; the environment adapter clips them to the
    bounds (the usual PPO treatment). The mean is expressed in units of the
    action range, so one initial scale suits every environment.

    Parameters
    ----------
    in_dim:
        Input size (observation, optionally with agent identifiers).
    action_dim:
        Action size.
    low, high:
        Bounds of shape ``(action_dim,)`` used to centre and scale the mean.
    hidden_sizes, activation:
        See :func:`mlp`.
    log_std_init:
        Initial log standard deviation, in units of half the action range.
    """

    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        low: np.ndarray,
        high: np.ndarray,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.net = mlp(in_dim, action_dim, hidden_sizes, activation, output_gain=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.register_buffer("center", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))
        self.register_buffer("half_range", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and standard deviation in action units."""
        mean = self.center + self.half_range * self.net(x)
        std = self.half_range * self.log_std.exp()
        return mean, std.expand_as(mean)

    def sample(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and their log-probabilities (summed over action dimensions)."""
        mean, std = self(x)
        noise = torch.randn(mean.shape, generator=generator, device=mean.device)
        action = mean + std * noise
        return action, self.log_prob_from(mean, std, action)

    def log_prob(self, x: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-probabilities of ``action`` and the entropy of the distribution."""
        mean, std = self(x)
        entropy = (0.5 + 0.5 * np.log(2.0 * np.pi) + std.log()).sum(-1)
        return self.log_prob_from(mean, std, action), entropy

    @staticmethod
    def log_prob_from(mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Diagonal Gaussian log-density summed over the last axis."""
        z = (action - mean) / std
        return (-0.5 * z.pow(2) - std.log() - 0.5 * np.log(2.0 * np.pi)).sum(-1)


class CategoricalPolicy(nn.Module):
    """Categorical policy over ``n_actions`` choices.

    Parameters
    ----------
    in_dim, n_actions:
        Input size and number of choices.
    hidden_sizes, activation:
        See :func:`mlp`.
    """

    def __init__(
        self,
        in_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.net = mlp(in_dim, n_actions, hidden_sizes, activation, output_gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits of shape ``(..., n_actions)``."""
        return self.net(x)

    def sample(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions (int64) and their log-probabilities."""
        logits = self(x)
        log_probs = F.log_softmax(logits, dim=-1)
        flat = log_probs.reshape(-1, log_probs.shape[-1]).exp()
        action = torch.multinomial(flat, 1, generator=generator).reshape(logits.shape[:-1])
        return action, log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, x: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-probabilities of ``action`` and the entropy of the distribution."""
        log_probs = F.log_softmax(self(x), dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        return log_probs.gather(-1, action.long().unsqueeze(-1)).squeeze(-1), entropy


class DeterministicPolicy(nn.Module):
    """Deterministic policy squashed into the action bounds with ``tanh``.

    Parameters
    ----------
    in_dim, action_dim:
        Input and action sizes.
    low, high:
        Bounds of shape ``(action_dim,)``.
    hidden_sizes, activation:
        See :func:`mlp`.
    """

    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        low: np.ndarray,
        high: np.ndarray,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.net = mlp(in_dim, action_dim, hidden_sizes, activation, output_gain=0.01)
        self.register_buffer("center", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))
        self.register_buffer("half_range", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Actions within the bounds."""
        return self.center + self.half_range * torch.tanh(self.net(x))


class QMixer(nn.Module):
    """Monotonic mixing network of QMIX.

    Combines per-agent utilities ``q`` of shape ``(..., n_agents)`` into a team
    value with state-conditioned non-negative weights (hypernetworks), so
    ``argmax`` of the team value decomposes into per-agent ``argmax``.

    Parameters
    ----------
    n_agents:
        Number of agents.
    state_dim:
        Size of the global state.
    embed_dim:
        Width of the mixing layer.
    hypernet_hidden:
        Width of the hypernetworks' hidden layer.
    """

    def __init__(
        self, n_agents: int, state_dim: int, embed_dim: int = 32, hypernet_hidden: int = 64
    ) -> None:
        super().__init__()
        self.n_agents, self.embed_dim = int(n_agents), int(embed_dim)
        self.hyper_w1 = mlp(state_dim, n_agents * embed_dim, (hypernet_hidden,), "relu")
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = mlp(state_dim, embed_dim, (hypernet_hidden,), "relu")
        self.hyper_b2 = mlp(state_dim, 1, (embed_dim,), "relu")

    def forward(self, q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Team value of shape ``q.shape[:-1]``."""
        batch = q.shape[:-1]
        q = q.reshape(-1, 1, self.n_agents)
        state = state.reshape(-1, state.shape[-1])
        w1 = self.hyper_w1(state).abs().view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(q, w1) + b1)
        w2 = self.hyper_w2(state).abs().view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(-1, 1, 1)
        return (torch.bmm(hidden, w2) + b2).reshape(batch)


class PerAgent(nn.Module):
    """One network shared by all agents, or one network per agent.

    Inputs have shape ``(*batch, n_agents, features)``. With ``shared=True`` the
    single network is applied to all agents at once; otherwise agent ``i``'s
    slice goes through network ``i`` and the outputs are stacked again on the
    agent axis.

    Parameters
    ----------
    factory:
        Callable returning a new network.
    n_agents:
        Number of agents.
    shared:
        Share one network between all agents.

    Examples
    --------
    >>> net = PerAgent(lambda: mlp(5, 2), n_agents=3, shared=False)
    >>> net(torch.zeros(4, 3, 5)).shape
    torch.Size([4, 3, 2])
    """

    def __init__(
        self, factory: Callable[[], nn.Module], n_agents: int, shared: bool = True
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.shared = bool(shared)
        self.nets = nn.ModuleList([factory() for _ in range(1 if shared else self.n_agents)])

    def forward(self, x: torch.Tensor, *args: torch.Tensor):
        return self.call("forward", x, *args)

    def call(self, method: str, x: torch.Tensor, *args: torch.Tensor):
        """Call ``method`` of the underlying network(s) on per-agent inputs.

        Every tensor argument must have the agent axis at the same position as
        ``x`` (``x.dim() - 2``); other arguments (such as a
        ``torch.Generator``) are passed unchanged. Outputs (tensors or tuples of
        tensors) are stacked on the agent axis.
        """
        if self.shared:
            return getattr(self.nets[0], method)(x, *args)
        axis = x.dim() - 2
        outputs = [
            getattr(net, method)(x.select(axis, i), *(_agent_slice(a, axis, i) for a in args))
            for i, net in enumerate(self.nets)
        ]
        if isinstance(outputs[0], tuple):
            return tuple(torch.stack(parts, dim=axis) for parts in zip(*outputs))
        return torch.stack(outputs, dim=axis)


def _agent_slice(value, axis: int, index: int):
    """Agent ``index`` of a per-agent tensor; non-tensors are returned unchanged."""
    return value.select(axis, index) if isinstance(value, torch.Tensor) else value
