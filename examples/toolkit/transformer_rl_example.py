"""Proximal Policy Optimisation (PPO) with transformer actor and critic networks.

A toy environment emits observation vectors; the agent sees the last
``seq_length`` of them as a sequence ``(seq_length, obs_dim)``. Action ``k`` in
``{0, 1, 2}`` pays ``1.0``, ``0.5`` or ``0.3`` when feature ``k`` of the current
observation is positive and ``-0.1`` otherwise (action 3 always pays ``-0.1``),
plus Gaussian noise. The chosen feature is then perturbed, so observations
evolve with the agent's actions.

:class:`toolkit.neural_toolkit.TransformerPolicyNetwork` (actor) and
:class:`toolkit.neural_toolkit.TransformerValueNetwork` (critic) are trained with
PPO: rollouts from several environments in lockstep, generalised advantage
estimation (GAE), a clipped surrogate objective, value regression and an entropy
bonus. Training curves are saved to ``renders/``.

Usage::

    python examples/toolkit/transformer_rl_example.py              # full run (about a minute on CPU)
    python examples/toolkit/transformer_rl_example.py --quick      # smoke test (a few seconds)
    python examples/toolkit/transformer_rl_example.py --iterations 100 --device cuda
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from toolkit.neural_toolkit import NetworkUtils, TransformerPolicyNetwork, TransformerValueNetwork

ACTION_REWARDS = (1.0, 0.5, 0.3)  # payout of actions 0, 1, 2 when their feature is positive


class SequentialEnvironment:
    """Toy environment with sequential observations (see the module docstring).

    Parameters
    ----------
    seq_length : int
        Length of the observation history returned to the agent.
    obs_dim : int
        Features per observation (at least ``action_dim``).
    action_dim : int
        Number of discrete actions.
    max_steps : int
        Episode length (episodes end by time limit only).
    seed : int, optional
        Seed of the environment's random generator.
    """

    def __init__(
        self,
        seq_length: int = 10,
        obs_dim: int = 8,
        action_dim: int = 4,
        max_steps: int = 64,
        seed: int | None = None,
    ) -> None:
        if obs_dim < action_dim:
            raise ValueError("obs_dim must be at least action_dim")
        self.seq_length = seq_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.history = np.zeros((seq_length, obs_dim), dtype=np.float32)
        self.step_count = 0

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.history = self.rng.standard_normal((self.seq_length, self.obs_dim)).astype(np.float32)
        return self.history.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        current = self.history[-1]
        if action < len(ACTION_REWARDS) and current[action] > 0:
            reward = ACTION_REWARDS[action]
        else:
            reward = -0.1
        reward += float(self.rng.normal(0.0, 0.1))
        new_obs = current.copy()
        new_obs[action] += self.rng.normal(0.0, 0.5)
        new_obs = np.clip(new_obs, -2.0, 2.0)
        self.history = np.concatenate([self.history[1:], new_obs[None]], axis=0)
        self.step_count += 1
        return self.history.copy(), reward, self.step_count >= self.max_steps


@dataclass
class Rollout:
    """Transitions of ``num_envs`` environments over ``T`` steps (time-major tensors)."""

    obs: torch.Tensor  # (T, N, seq_length, obs_dim)
    actions: torch.Tensor  # (T, N)
    log_probs: torch.Tensor  # (T, N)
    values: torch.Tensor  # (T, N)
    rewards: torch.Tensor  # (T, N)


class TransformerPPOAgent:
    """PPO agent with transformer actor and critic.

    Parameters
    ----------
    obs_dim, action_dim : int
        Observation features and number of actions.
    d_model, nhead, num_layers, dim_feedforward, dropout
        Transformer hyperparameters shared by actor and critic.
    lr : float
        Adam learning rate.
    device : str
        Torch device; the networks are created directly on it.
    seed : int
        Seed of the agent's minibatch shuffling generator.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        lr: float = 3e-4,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        common = dict(
            input_dim=obs_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            fc_dims=(d_model // 2,),
            device=device,
        )
        self.device = torch.device(device)
        self.policy = TransformerPolicyNetwork(output_dim=action_dim, **common)
        self.value = TransformerValueNetwork(output_dim=1, **common)
        for net in (self.policy, self.value):
            NetworkUtils.initialize_weights(net, method="orthogonal", gain=np.sqrt(2.0))
        # Small initial logits give a near-uniform starting policy.
        NetworkUtils.initialize_weights(self.policy.output_layer, method="orthogonal", gain=0.01)
        NetworkUtils.initialize_weights(self.value.output_layer, method="orthogonal", gain=1.0)
        self.optimizer = torch.optim.Adam(
            [*self.policy.parameters(), *self.value.parameters()], lr=lr, eps=1e-5
        )
        self.generator = torch.Generator().manual_seed(seed)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

    def _obs_tensor(self, obs: Sequence[np.ndarray]) -> torch.Tensor:
        return torch.as_tensor(np.stack(obs), device=self.device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, greedy: bool = False) -> tuple[torch.Tensor, ...]:
        """Return ``(actions, log_probs, values)`` for a batch of observation sequences."""
        dist = torch.distributions.Categorical(logits=self.policy(obs))
        actions = dist.probs.argmax(dim=-1) if greedy else dist.sample()
        return actions, dist.log_prob(actions), self.value(obs).squeeze(-1)

    def collect(self, envs: list[SequentialEnvironment]) -> tuple[Rollout, np.ndarray]:
        """Run one episode in every environment (lockstep) and return the rollout and returns."""
        self.policy.eval()
        self.value.eval()
        obs = [env.reset() for env in envs]
        steps: dict[str, list[torch.Tensor]] = {k: [] for k in Rollout.__dataclass_fields__}
        done = False
        while not done:
            obs_t = self._obs_tensor(obs)
            actions, log_probs, values = self.act(obs_t)
            results = [env.step(int(a)) for env, a in zip(envs, actions.tolist())]
            obs = [r[0] for r in results]
            rewards = torch.tensor([r[1] for r in results], dtype=torch.float32, device=self.device)
            done = all(r[2] for r in results)
            for key, value in zip(
                ("obs", "actions", "log_probs", "values", "rewards"),
                (obs_t, actions, log_probs, values, rewards),
            ):
                steps[key].append(value)
        rollout = Rollout(**{k: torch.stack(v) for k, v in steps.items()})
        return rollout, rollout.rewards.sum(dim=0).cpu().numpy()

    def advantages(self, rollout: Rollout) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE advantages and value targets; the time limit is treated as terminal."""
        rewards, values = rollout.rewards, rollout.values
        advantages = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[0])
        for t in reversed(range(rewards.shape[0])):
            next_value = values[t + 1] if t + 1 < rewards.shape[0] else torch.zeros_like(running)
            delta = rewards[t] + self.gamma * next_value - values[t]
            running = delta + self.gamma * self.gae_lambda * running
            advantages[t] = running
        return advantages, advantages + values

    def update(
        self, rollout: Rollout, epochs: int = 4, minibatch_size: int = 64
    ) -> dict[str, float]:
        """Run PPO epochs over the rollout and return mean losses."""
        advantages, returns = self.advantages(rollout)
        obs = rollout.obs.flatten(0, 1)
        actions = rollout.actions.flatten()
        old_log_probs = rollout.log_probs.flatten()
        advantages = advantages.flatten()
        returns = returns.flatten()
        self.policy.train()
        self.value.train()
        stats: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": []}
        n = obs.shape[0]
        for _ in range(epochs):
            order = torch.randperm(n, generator=self.generator).to(self.device)
            for start in range(0, n, minibatch_size):
                idx = order[start : start + minibatch_size]
                adv = advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(idx) > 1 else adv
                dist = torch.distributions.Categorical(logits=self.policy(obs[idx]))
                ratio = torch.exp(dist.log_prob(actions[idx]) - old_log_probs[idx])
                clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                value_loss = F.mse_loss(self.value(obs[idx]).squeeze(-1), returns[idx])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                NetworkUtils.clip_grad_norm(self.policy, self.max_grad_norm)
                NetworkUtils.clip_grad_norm(self.value, self.max_grad_norm)
                self.optimizer.step()
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
        return {key: float(np.mean(values)) for key, values in stats.items()}

    def evaluate(self, envs: list[SequentialEnvironment]) -> np.ndarray:
        """Greedy episode returns, one per environment."""
        self.policy.eval()
        obs = [env.reset() for env in envs]
        totals = np.zeros(len(envs))
        done = False
        while not done:
            actions, _, _ = self.act(self._obs_tensor(obs), greedy=True)
            results = [env.step(int(a)) for env, a in zip(envs, actions.tolist())]
            obs = [r[0] for r in results]
            totals += [r[1] for r in results]
            done = all(r[2] for r in results)
        return totals


def random_policy_return(envs: list[SequentialEnvironment], rng: np.random.Generator) -> float:
    """Mean return of a uniformly random policy (reference level)."""
    totals = []
    for env in envs:
        env.reset()
        total, done = 0.0, False
        while not done:
            _, reward, done = env.step(int(rng.integers(env.action_dim)))
            total += reward
        totals.append(total)
    return float(np.mean(totals))


def save_training_plots(history: dict[str, list[float]], eval_x: list[int], path: Path) -> None:
    """Save a 2x2 panel of training curves (no GUI backend needed)."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 7), dpi=100, layout="constrained")
    fig.suptitle("Transformer PPO training")
    axes = fig.subplots(2, 2)
    panels = [
        (axes[0, 0], history["train_return"], None, "Training return (mean over envs)", "tab:blue"),
        (axes[0, 1], history["eval_return"], eval_x, "Greedy evaluation return", "tab:red"),
        (axes[1, 0], history["policy_loss"], None, "Policy loss", "tab:green"),
        (axes[1, 1], history["value_loss"], None, "Value loss", "tab:orange"),
    ]
    for ax, values, x, title, color in panels:
        xs = x if x is not None else range(1, len(values) + 1)
        ax.plot(xs, values, marker="o" if x is not None else None, color=color, linewidth=1.5)
        ax.set(title=title, xlabel="Iteration")
        ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--iterations", type=int, default=40, help="PPO iterations")
    cli.add_argument("--num-envs", type=int, default=8, help="parallel environments per rollout")
    cli.add_argument("--max-steps", type=int, default=64, help="episode length")
    cli.add_argument("--seq-length", type=int, default=10, help="observation history length")
    cli.add_argument("--eval-every", type=int, default=5, help="evaluation interval (iterations)")
    cli.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    cli.add_argument("--seed", type=int, default=0, help="random seed")
    cli.add_argument("--device", default="auto", help="'auto', 'cpu' or 'cuda'")
    cli.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    cli.add_argument(
        "--save",
        type=Path,
        default=Path("renders/transformer_rl_training.png"),
        help="training-curve image",
    )
    cli.add_argument("--no-plot", action="store_true", help="do not save the training curves")
    args = cli.parse_args(argv)
    if args.quick:
        args.iterations, args.num_envs, args.max_steps, args.eval_every = 2, 2, 16, 1
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    torch.manual_seed(args.seed)
    env_kwargs = dict(seq_length=args.seq_length, obs_dim=8, action_dim=4, max_steps=args.max_steps)
    train_envs = [
        SequentialEnvironment(**env_kwargs, seed=args.seed + i) for i in range(args.num_envs)
    ]
    eval_envs = [
        SequentialEnvironment(**env_kwargs, seed=10_000 + args.seed + i)
        for i in range(args.num_envs)
    ]
    agent = TransformerPPOAgent(obs_dim=8, action_dim=4, lr=args.lr, device=device, seed=args.seed)
    baseline = random_policy_return(eval_envs, np.random.default_rng(args.seed))
    n_params = NetworkUtils.count_parameters(agent.policy) + NetworkUtils.count_parameters(
        agent.value
    )
    print(
        f"PPO on {args.num_envs} envs x {args.max_steps} steps, {args.iterations} iterations, "
        f"device {device}, {n_params:,} parameters"
    )
    print(f"Random-policy return: {baseline:.2f}")

    history: dict[str, list[float]] = {
        k: [] for k in ("train_return", "eval_return", "policy_loss", "value_loss")
    }
    eval_x: list[int] = []
    start = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        rollout, returns = agent.collect(train_envs)
        stats = agent.update(rollout)
        history["train_return"].append(float(returns.mean()))
        history["policy_loss"].append(stats["policy_loss"])
        history["value_loss"].append(stats["value_loss"])
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            scores = agent.evaluate(eval_envs)
            history["eval_return"].append(float(scores.mean()))
            eval_x.append(iteration)
            print(
                f"iter {iteration:3d}  train {returns.mean():6.2f}  eval {scores.mean():6.2f} "
                f"+/- {scores.std():5.2f}  policy loss {stats['policy_loss']:7.4f}  "
                f"value loss {stats['value_loss']:7.4f}  entropy {stats['entropy']:.3f}"
            )
    elapsed = time.perf_counter() - start

    final = agent.evaluate(eval_envs)
    print(f"Training time: {elapsed:.1f} s")
    print(
        f"Final greedy return: {final.mean():.2f} +/- {final.std():.2f} "
        f"(random policy {baseline:.2f})"
    )
    obs = torch.as_tensor(eval_envs[0].reset()[None], device=agent.device)
    actions = [int(agent.act(obs, greedy=True)[0])]
    for _ in range(9):
        next_obs, _, _ = eval_envs[0].step(actions[-1])
        obs = torch.as_tensor(next_obs[None], device=agent.device)
        actions.append(int(agent.act(obs, greedy=True)[0]))
    print(f"First greedy actions of an evaluation episode: {actions}")

    if not args.no_plot:
        save_training_plots(history, eval_x, args.save)
        print(f"Training curves saved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
