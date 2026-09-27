"""Paper-style summary figure of reinforcement-learning results with plotkit.

The script builds a three-panel figure from synthetic data:

(a) learning curves of three methods over several seeds with 95 % confidence bands
    and EMA smoothing (``plot_learning_curves``);
(b) grouped bars of the final return per environment with 95 % CI error bars
    (``plot_bar``);
(c) a heatmap of a learning-rate x batch-size sweep with annotated cells
    (``plot_heatmap``).

Each method keeps the same colour-blind-safe (Okabe-Ito) colour in every panel. The
figure is saved to ``renders/plotkit_gallery.pdf`` and ``renders/plotkit_gallery.png``.
All numbers are generated from ``--seed`` and do not describe real algorithms.

Run::

    python examples/plotkit_gallery.py
    python examples/plotkit_gallery.py --style presentation --save renders/gallery_talk --show
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from toolkit.plotkit import (
    STYLE_PRESETS,
    get_palette,
    plot_bar,
    plot_heatmap,
    plot_learning_curves,
    save_figure,
    style_context,
)

METHODS = {  # name: (learning speed, asymptotic return)
    "Baseline": (0.010, 70.0),
    "Method A": (0.018, 92.0),
    "Method B": (0.030, 80.0),
}
ENVIRONMENTS = ["Env 1", "Env 2", "Env 3", "Env 4"]
LEARNING_RATES = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]
BATCH_SIZES = ["32", "64", "128", "256"]


def synthetic_runs(rng: np.random.Generator, n_seeds: int, n_steps: int) -> dict[str, np.ndarray]:
    """Learning curves of shape (n_seeds, n_steps) per method."""
    steps = np.arange(n_steps)
    runs = {}
    for name, (speed, final) in METHODS.items():
        scale = final * rng.uniform(0.9, 1.1, size=(n_seeds, 1))
        curve = scale * (1.0 - np.exp(-speed * steps * 200.0 / n_steps))
        runs[name] = curve + rng.normal(0.0, 6.0, size=(n_seeds, n_steps))
    return runs


def synthetic_final_returns(rng: np.random.Generator, n_seeds: int) -> np.ndarray:
    """Final returns of shape (n_methods, n_envs, n_seeds)."""
    finals = np.array([final for _, final in METHODS.values()])[:, None, None]
    difficulty = np.array([1.0, 0.8, 0.65, 0.9])[None, :, None]
    return finals * difficulty + rng.normal(
        0.0, 5.0, size=(len(METHODS), len(ENVIRONMENTS), n_seeds)
    )


def synthetic_sweep(rng: np.random.Generator) -> np.ndarray:
    """Final return for each (learning rate, batch size) pair."""
    lr = np.arange(len(LEARNING_RATES))[:, None]
    bs = np.arange(len(BATCH_SIZES))[None, :]
    surface = 90.0 * np.exp(-0.5 * ((lr - 2.0) / 1.2) ** 2 - 0.5 * ((bs - 1.5) / 1.8) ** 2)
    return surface + rng.normal(0.0, 2.0, size=surface.shape)


def build_figure(seed: int = 0, n_seeds: int = 8, n_steps: int = 200, style: str = "research"):
    """Create the three-panel figure and return it."""
    rng = np.random.default_rng(seed)
    colors = get_palette("okabe_ito", len(METHODS))
    runs = synthetic_runs(rng, n_seeds, n_steps)
    finals = synthetic_final_returns(rng, n_seeds)
    sweep = synthetic_sweep(rng)

    with style_context(style):
        fig, axes = plt.subplots(
            1, 3, figsize=(16, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1, 1]}
        )
        plot_learning_curves(
            runs,
            x=np.linspace(0, 1000, n_steps),
            band="ci95",
            smoothing=0.6,
            colors=colors,
            ax=axes[0],
            title="(a) Learning curves",
            xlabel="Environment steps (thousands)",
            ylabel="Episode return",
            style=style,
        )
        ci95 = 1.96 * finals.std(axis=2, ddof=1) / np.sqrt(n_seeds)
        plot_bar(
            ENVIRONMENTS,
            finals.mean(axis=2),
            labels=list(METHODS),
            colors=colors,
            yerr=ci95,
            ax=axes[1],
            title="(b) Final return",
            ylabel="Return (mean, 95% CI)",
            legend=False,
            style=style,
        )
        plot_heatmap(
            sweep,
            xlabels=BATCH_SIZES,
            ylabels=LEARNING_RATES,
            annot=True,
            fmt=".0f",
            cbar_label="Final return",
            ax=axes[2],
            title="(c) Hyperparameter sweep",
            xlabel="Batch size",
            ylabel="Learning rate",
            style=style,
        )
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--seeds", type=int, default=8, help="training seeds per method (default: 8)"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="points per learning curve (default: 200)"
    )
    parser.add_argument(
        "--style",
        default="research",
        choices=list(STYLE_PRESETS),
        help="style preset (default: research)",
    )
    parser.add_argument(
        "--save",
        default="renders/plotkit_gallery",
        help="output path without extension (default: renders/plotkit_gallery)",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["pdf", "png"], help="output formats (default: pdf png)"
    )
    parser.add_argument("--dpi", type=float, default=300, help="raster resolution (default: 300)")
    parser.add_argument("--show", action="store_true", help="also show the figure in a window")
    args = parser.parse_args(argv)

    fig = build_figure(seed=args.seed, n_seeds=args.seeds, n_steps=args.steps, style=args.style)
    for path in save_figure(fig, args.save, formats=args.formats, dpi=args.dpi):
        print(f"saved {path}")
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
