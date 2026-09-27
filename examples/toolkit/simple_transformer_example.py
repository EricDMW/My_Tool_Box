"""Train a transformer policy with REINFORCE on a small memory task.

The environment shows the agent a sliding window of the last ``seq_length``
observation vectors. The rewarded action is the index of the largest of the first
``action_dim`` features of the *oldest* observation in the window, so the policy
has to use positional information: a permutation-invariant model cannot solve
the task. A random policy earns ``1 / action_dim`` per step on average.

The script trains :class:`toolkit.neural_toolkit.TransformerPolicyNetwork` with
REINFORCE (normalised immediate rewards as advantages plus an entropy bonus),
evaluates the greedy policy, compares parameter counts with an MLP on the
flattened window, and saves a learning curve to ``renders/``.

Usage::

    python examples/toolkit/simple_transformer_example.py              # full run (about a minute on CPU)
    python examples/toolkit/simple_transformer_example.py --quick      # smoke test (a few seconds)
    python examples/toolkit/simple_transformer_example.py --episodes 500 --save renders/curve.png
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from toolkit.neural_toolkit import MLPPolicyNetwork, NetworkUtils, TransformerPolicyNetwork


class MemoryEnv:
    """Sliding-window observation task (see the module docstring).

    Parameters
    ----------
    seq_length : int
        Number of observations in the window.
    obs_dim : int
        Features per observation (must be >= ``action_dim``).
    action_dim : int
        Number of discrete actions.
    max_steps : int
        Episode length.
    seed : int, optional
        Seed of the environment's random generator.
    """

    def __init__(
        self,
        seq_length: int = 5,
        obs_dim: int = 4,
        action_dim: int = 3,
        max_steps: int = 20,
        seed: int | None = None,
    ) -> None:
        if obs_dim < action_dim:
            raise ValueError("obs_dim must be at least action_dim")
        self.seq_length = seq_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.window = np.zeros((seq_length, obs_dim), dtype=np.float32)
        self.step_count = 0

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.window = self.rng.standard_normal((self.seq_length, self.obs_dim)).astype(np.float32)
        return self.window.copy()

    def target_action(self) -> int:
        return int(np.argmax(self.window[0, : self.action_dim]))

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        reward = 1.0 if action == self.target_action() else 0.0
        new_obs = self.rng.standard_normal(self.obs_dim).astype(np.float32)
        self.window = np.concatenate([self.window[1:], new_obs[None]], axis=0)
        self.step_count += 1
        return self.window.copy(), reward, self.step_count >= self.max_steps


def run_episode(
    policy: torch.nn.Module, env: MemoryEnv, greedy: bool = False
) -> tuple[float, list[torch.Tensor], list[torch.Tensor], list[float]]:
    """Play one episode; returns (return, log-probs, entropies, rewards)."""
    obs = env.reset()
    log_probs, entropies, rewards = [], [], []
    done = False
    while not done:
        logits = policy(torch.as_tensor(obs).unsqueeze(0))  # (1, action_dim)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if greedy else dist.sample()
        obs, reward, done = env.step(int(action.item()))
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        rewards.append(reward)
    return float(sum(rewards)), log_probs, entropies, rewards


def train(
    policy: torch.nn.Module,
    env: MemoryEnv,
    episodes: int,
    lr: float,
    entropy_coef: float = 0.01,
    log_every: int = 25,
) -> list[float]:
    """REINFORCE on immediate rewards (each step is an independent decision)."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    returns: list[float] = []
    policy.train()
    for episode in range(1, episodes + 1):
        total, log_probs, entropies, rewards = run_episode(policy, env)
        advantage = torch.tensor(rewards)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        loss = (
            -(torch.cat(log_probs) * advantage).mean() - entropy_coef * torch.cat(entropies).mean()
        )
        optimizer.zero_grad()
        loss.backward()
        NetworkUtils.clip_grad_norm(policy, max_norm=1.0)
        optimizer.step()
        returns.append(total)
        if episode % log_every == 0 or episode == episodes:
            recent = np.mean(returns[-log_every:])
            print(
                f"episode {episode:4d}  return {total:5.1f}  mean(last {log_every}) {recent:5.2f}"
            )
    return returns


def evaluate(policy: torch.nn.Module, env: MemoryEnv, episodes: int) -> np.ndarray:
    policy.eval()
    with torch.no_grad():
        return np.array([run_episode(policy, env, greedy=True)[0] for _ in range(episodes)])


def save_learning_curve(returns: Sequence[float], window: int, path: Path) -> None:
    """Save episode returns and their moving average (no GUI backend needed)."""
    from matplotlib.figure import Figure

    values = np.asarray(returns, dtype=float)
    kernel = np.ones(min(window, len(values))) / min(window, len(values))
    smooth = np.convolve(values, kernel, mode="valid")
    fig = Figure(figsize=(10, 4.5), dpi=100, layout="constrained")
    ax_raw, ax_avg = fig.subplots(1, 2)
    ax_raw.plot(values, color="tab:blue", alpha=0.6, linewidth=1)
    ax_raw.set(title="Episode return", xlabel="Episode", ylabel="Return")
    ax_avg.plot(np.arange(len(smooth)) + len(kernel), smooth, color="tab:red", linewidth=2)
    ax_avg.set(title=f"{len(kernel)}-episode moving average", xlabel="Episode", ylabel="Return")
    for ax in (ax_raw, ax_avg):
        ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--episodes", type=int, default=300, help="training episodes")
    cli.add_argument("--eval-episodes", type=int, default=20, help="greedy evaluation episodes")
    cli.add_argument("--lr", type=float, default=3e-3, help="Adam learning rate")
    cli.add_argument("--seed", type=int, default=0, help="random seed")
    cli.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    cli.add_argument(
        "--save",
        type=Path,
        default=Path("renders/simple_transformer_training.png"),
        help="learning-curve image",
    )
    cli.add_argument("--no-plot", action="store_true", help="do not save the learning curve")
    args = cli.parse_args(argv)
    if args.quick:
        args.episodes, args.eval_episodes = 5, 2

    torch.manual_seed(args.seed)
    env = MemoryEnv(seq_length=5, obs_dim=4, action_dim=3, max_steps=20, seed=args.seed)
    policy = TransformerPolicyNetwork(
        input_dim=env.obs_dim,
        output_dim=env.action_dim,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        fc_dims=(32,),
        device="cpu",
    )
    mlp = MLPPolicyNetwork(env.seq_length * env.obs_dim, env.action_dim, hidden_dims=(64, 64))
    print(
        f"Memory task: window {env.seq_length}x{env.obs_dim}, {env.action_dim} actions, "
        f"{env.max_steps} steps per episode (random policy: {env.max_steps / env.action_dim:.1f})"
    )
    print(
        f"Transformer policy parameters: {NetworkUtils.count_parameters(policy):,} "
        f"(MLP on the flattened window: {NetworkUtils.count_parameters(mlp):,})"
    )

    start = time.perf_counter()
    returns = train(policy, env, args.episodes, args.lr, log_every=max(1, min(25, args.episodes)))
    elapsed = time.perf_counter() - start
    scores = evaluate(policy, env, args.eval_episodes)
    print(f"Training time: {elapsed:.1f} s")
    print(
        f"Greedy evaluation over {len(scores)} episodes: mean {scores.mean():.2f}, "
        f"std {scores.std():.2f}, max {env.max_steps}"
    )

    sample = torch.as_tensor(env.reset()).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(policy(sample), dim=-1).squeeze(0).numpy()
    print(f"Sample decision: target {env.target_action()}, probabilities {np.round(probs, 3)}")

    if not args.no_plot:
        save_learning_curve(returns, window=25, path=args.save)
        print(f"Learning curve saved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
