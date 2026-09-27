"""Command-line gallery of plotkit plots.

Examples
--------
Show every plot type in one window (or save it when the backend is non-interactive)::

    python -m toolkit.plotkit --demo all

Save a single demo in two formats with the presentation style::

    plotkit-gallery --demo heatmap --style presentation --save renders/plotkit --format png pdf

Without ``--save`` the figure is shown with :func:`matplotlib.pyplot.show`; on a
non-interactive backend (e.g. ``MPLBACKEND=Agg``) it is saved to ``renders/plotkit``
instead. All demo data are synthetic and generated from ``--seed``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .io import save_figure
from .plots import (
    plot_bar,
    plot_gray_scale,
    plot_heatmap,
    plot_histogram,
    plot_learning_curves,
    plot_line,
    plot_scatter,
    plot_shadow_curve,
)
from .styles import STYLE_PRESETS, style_context

DEFAULT_SAVE_DIR = Path("renders") / "plotkit"


def _demo_shadow(ax: Axes, rng: np.random.Generator, style: str) -> None:
    x = np.linspace(0, 10, 100)
    sine = np.sin(x) + rng.normal(0, 0.25, (8, x.size))
    cosine = np.cos(x) + rng.normal(0, 0.25, (8, x.size))
    plot_shadow_curve(
        [sine, cosine],
        x=x,
        labels=["Sine", "Cosine"],
        band="std",
        title="Shadow curve (mean +/- std)",
        xlabel="Time",
        ylabel="Amplitude",
        ax=ax,
        style=style,
    )


def _demo_learning(ax: Axes, rng: np.random.Generator, style: str) -> None:
    steps = np.arange(200)
    runs = {}
    for name, rate, final in (("Baseline", 0.015, 80.0), ("Method A", 0.03, 100.0)):
        curve = final * (1 - np.exp(-rate * steps))
        runs[name] = curve + rng.normal(0, 8.0, (6, steps.size))
    plot_learning_curves(
        runs,
        x=steps,
        band="ci95",
        smoothing=0.8,
        title="Learning curves (95% CI, EMA 0.8)",
        xlabel="Environment steps (thousands)",
        ylabel="Episode return",
        ax=ax,
        style=style,
    )


def _demo_heatmap(ax: Axes, rng: np.random.Generator, style: str) -> None:
    lrs = ["1e-4", "3e-4", "1e-3", "3e-3"]
    batches = ["32", "64", "128", "256", "512"]
    grid = np.array(
        [[40, 55, 60, 58, 50], [62, 78, 85, 80, 70], [58, 80, 92, 88, 75], [30, 45, 50, 52, 48]]
    )
    plot_heatmap(
        grid + rng.normal(0, 2.0, grid.shape),
        xlabels=batches,
        ylabels=lrs,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        cbar_label="Final return",
        title="Hyperparameter sweep",
        xlabel="Batch size",
        ylabel="Learning rate",
        ax=ax,
        style=style,
    )


def _demo_grayscale(ax: Axes, rng: np.random.Generator, style: str) -> None:
    yy, xx = np.mgrid[-2:2:24j, -2:2:24j]
    image = np.exp(-(xx**2 + yy**2)) + rng.normal(0, 0.05, xx.shape)
    plot_gray_scale(
        image,
        xlabels=False,
        ylabels=False,
        linewidths=0,
        title="Grayscale image",
        ax=ax,
        style=style,
    )


def _demo_line(ax: Axes, rng: np.random.Generator, style: str) -> None:
    t = np.linspace(0, 2 * np.pi, 200)
    plot_line(
        t,
        [np.sin(t), np.cos(t), np.sin(2 * t) / 2],
        labels=["sin(t)", "cos(t)", "sin(2t) / 2"],
        title="Line plot (shared x)",
        xlabel="t",
        ylabel="Value",
        ax=ax,
        style=style,
    )


def _demo_bar(ax: Axes, rng: np.random.Generator, style: str) -> None:
    envs = ["Env 1", "Env 2", "Env 3", "Env 4"]
    means = np.array([[62, 75, 58, 80], [70, 82, 66, 85], [55, 60, 52, 71]], dtype=float)
    stds = rng.uniform(2, 6, means.shape)
    plot_bar(
        envs,
        means,
        labels=["Baseline", "Method A", "Method B"],
        yerr=stds,
        value_labels=".0f",
        title="Grouped bars with error bars",
        ylabel="Final return",
        ax=ax,
        style=style,
    )


def _demo_scatter(ax: Axes, rng: np.random.Generator, style: str) -> None:
    x = rng.normal(0, 1, 80)
    groups = [x + rng.normal(0, 0.4, 80), -0.5 * x + rng.normal(1.5, 0.4, 80)]
    plot_scatter(
        x,
        groups,
        labels=["Group A", "Group B"],
        title="Scatter (shared x)",
        xlabel="Feature",
        ylabel="Response",
        ax=ax,
        style=style,
    )


def _demo_histogram(ax: Axes, rng: np.random.Generator, style: str) -> None:
    plot_histogram(
        [rng.normal(0, 1, 1000), rng.normal(1.5, 0.7, 1000)],
        labels=["Policy A", "Policy B"],
        density=True,
        title="Return distribution",
        xlabel="Return",
        ax=ax,
        style=style,
    )


DEMOS: dict[str, Callable[[Axes, np.random.Generator, str], None]] = {
    "shadow": _demo_shadow,
    "learning": _demo_learning,
    "heatmap": _demo_heatmap,
    "grayscale": _demo_grayscale,
    "line": _demo_line,
    "bar": _demo_bar,
    "scatter": _demo_scatter,
    "histogram": _demo_histogram,
}


def build_figure(demo: str, style: str = "research", seed: int = 0) -> plt.Figure:
    """Create the figure of one demo, or of all demos in a grid for ``demo="all"``.

    Parameters
    ----------
    demo : str
        A key of :data:`DEMOS` or ``"all"``.
    style : str, default "research"
        Style preset (see :data:`toolkit.plotkit.STYLE_PRESETS`).
    seed : int, default 0
        Seed of the random generator used for the synthetic data.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if demo != "all" and demo not in DEMOS:
        raise ValueError(f"unknown demo {demo!r}; choose from {sorted(DEMOS)} or 'all'")
    rng = np.random.default_rng(seed)
    with style_context(style):
        if demo == "all":
            fig, axes = plt.subplots(2, 4, figsize=(24, 10), layout="constrained")
            for ax, draw in zip(axes.flat, DEMOS.values()):
                draw(ax, rng, style)
        else:
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            DEMOS[demo](ax, rng, style)
    return fig


def _backend_is_interactive() -> bool:
    backend = matplotlib.get_backend().lower()
    try:
        from matplotlib.backends import BackendFilter, backend_registry

        non_interactive = backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)
    except ImportError:  # matplotlib < 3.9
        from matplotlib import rcsetup

        non_interactive = rcsetup.non_interactive_bk
    return backend not in {name.lower() for name in non_interactive}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m toolkit.plotkit",
        description="Gallery of plotkit plots with synthetic data.",
    )
    parser.add_argument(
        "--demo",
        default="all",
        choices=[*DEMOS, "all"],
        help="plot type to show; 'all' draws every demo in one figure (default: all)",
    )
    parser.add_argument(
        "--style",
        default="research",
        choices=list(STYLE_PRESETS),
        help="style preset (default: research)",
    )
    parser.add_argument(
        "--save",
        metavar="DIR",
        type=Path,
        default=None,
        help="save the figure to DIR/plotkit_<demo>.<format> instead of showing it",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["png"],
        help="file format(s) used with --save (default: png)",
    )
    parser.add_argument(
        "--dpi", type=float, default=150, help="resolution of raster output (default: 150)"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gallery CLI.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit status (0 on success).
    """
    args = _parse_args(None if argv is None else [str(a) for a in argv])
    fig = build_figure(args.demo, style=args.style, seed=args.seed)
    save_dir = args.save
    if save_dir is None and not _backend_is_interactive():
        save_dir = DEFAULT_SAVE_DIR
    if save_dir is None:
        plt.show()
        return 0
    written: list[Path] = save_figure(
        fig, Path(save_dir) / f"plotkit_{args.demo}", formats=args.format, dpi=args.dpi
    )
    plt.close(fig)
    for path in written:
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
