"""Plot AJLATT occupancy maps.

Command line::

    ajlatt-plot-map obstacles04 --save obstacles04.png
    ajlatt-plot-map --list
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Union

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from env_lib.ajlatt_env.maps import GridMap, available_maps, load_grid_map
from env_lib.utils.rendering import Theme, get_theme, style_axes

__all__ = ["main", "plot_map"]

PathLike = Union[str, Path]


def plot_map(
    map_or_name: GridMap | PathLike,
    ax: Axes | None = None,
    *,
    theme: str | Theme | None = None,
    title: str | None = None,
    save: PathLike | None = None,
    dpi: int = 150,
) -> Axes:
    """Draw an occupancy map.

    Parameters
    ----------
    map_or_name:
        A :class:`GridMap`, a bundled map name or a path to a map header.
    ax:
        Axes to draw on (a new off-screen figure is created if ``None``).
    theme:
        Colour theme (defaults to the active theme).
    title:
        Axes title (defaults to the map name and size).
    save:
        Optional output path; the figure is written with ``dpi``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    theme = get_theme(theme)
    grid = map_or_name if isinstance(map_or_name, GridMap) else load_grid_map(map_or_name)
    if (
        getattr(grid, "map", None) is None
        and hasattr(grid, "generate_map")
        and not isinstance(map_or_name, GridMap)
    ):
        grid.generate_map(rng=np.random.default_rng(0))
    if ax is None:
        fig = Figure(figsize=(6.4, 6.4), facecolor=theme.background)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
    nx, ny = grid.mapdim
    occupancy = (
        np.zeros((ny, nx)) if grid.map is None else (np.asarray(grid.map) == 1).astype(float)
    )
    free = mcolors.to_rgb(theme.panel)
    wall = tuple(0.78 * np.array(free) + 0.22 * np.array(mcolors.to_rgb(theme.text)))
    ax.imshow(
        occupancy,
        cmap=mcolors.ListedColormap([free, wall]),
        vmin=0,
        vmax=1,
        origin="lower",
        interpolation="nearest",
        extent=[grid.mapmin[0], grid.mapmax[0], grid.mapmin[1], grid.mapmax[1]],
    )
    if title is None:
        width, height = grid.mapmax - grid.mapmin
        title = f"{grid.name}  ({width:g} m x {height:g} m, {grid.mapres[0]:g} m cells)"
    style_axes(ax, theme, title=title, xlabel="x [m]", ylabel="y [m]", grid=False, show_spines=True)
    ax.set_aspect("equal")
    if save is not None:
        ax.figure.savefig(save, dpi=dpi, bbox_inches="tight", facecolor=ax.figure.get_facecolor())
    return ax


def main(argv: Iterable[str] | None = None) -> int:
    """Command-line entry point (``ajlatt-plot-map``)."""
    parser = argparse.ArgumentParser(
        prog="ajlatt-plot-map", description="Plot an AJLATT occupancy map."
    )
    parser.add_argument(
        "map", nargs="?", default="obstacles04", help="bundled map name or map path"
    )
    parser.add_argument("--save", help="output image (default: <map>.png)")
    parser.add_argument("--theme", default="light", help="colour theme (dark or light)")
    parser.add_argument("--list", action="store_true", help="list the bundled maps and exit")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.list:
        print("\n".join(available_maps()))
        return 0
    output = Path(args.save) if args.save else Path(f"{Path(args.map).stem}.png")
    plot_map(args.map, theme=args.theme, save=output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
