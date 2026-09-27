"""Occupancy maps for the AJLATT environment.

Bundled maps live in ``maps/data`` as ``<name>.yaml`` (header) plus
``<name>.cfg`` (occupancy grid). Use :func:`available_maps` to list them and
:func:`load_grid_map` to construct a :class:`GridMap` (or :class:`DynamicMap`)
from a bundled name or from a path to your own map files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import yaml

from env_lib.ajlatt_env.maps.dynamic_map import DynamicMap
from env_lib.ajlatt_env.maps.grid_map import (
    GridMap,
    bresenham2D,
    bresenham_batch,
    cell_to_se2,
    coord_change2g,
    se2_to_cell,
)

__all__ = [
    "MAP_DIR",
    "DynamicMap",
    "GridMap",
    "available_maps",
    "bresenham2D",
    "bresenham_batch",
    "cell_to_se2",
    "coord_change2g",
    "load_grid_map",
    "load_map",
    "resolve_map_path",
    "se2_to_cell",
]

MAP_DIR = Path(__file__).resolve().parent / "data"

PathLike = Union[str, Path]


def available_maps() -> list[str]:
    """Names of the bundled maps that can be passed as ``map_name``."""
    names = []
    for header in sorted(MAP_DIR.glob("*.yaml")):
        with open(header, encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
        if "mapdim" not in content:
            continue
        if header.with_suffix(".cfg").exists() or "submaporigin" in content:
            names.append(header.stem)
    return names


def resolve_map_path(map_name: PathLike) -> Path:
    """Return the path (without extension) of a bundled or user-supplied map.

    ``map_name`` may be a bundled name (``"obstacles04"``) or a path to
    ``my_map`` / ``my_map.yaml`` whose header sits next to its ``.cfg`` file.
    """
    candidate = Path(map_name)
    if candidate.suffix in {".yaml", ".cfg"}:
        candidate = candidate.with_suffix("")
    if candidate.with_name(candidate.name + ".yaml").exists():
        return candidate
    bundled = MAP_DIR / candidate.name
    if bundled.with_name(bundled.name + ".yaml").exists():
        return bundled
    raise FileNotFoundError(
        f"Unknown map {str(map_name)!r}. Bundled maps: {', '.join(available_maps())}"
    )


def load_grid_map(map_name: PathLike, margin2wall: float = 0.5) -> GridMap:
    """Load a map by bundled name or path.

    A map with an occupancy file (``<name>.cfg``) is static (:class:`GridMap`).
    A header without one that defines ``submaporigin`` produces a
    :class:`DynamicMap` (call ``generate_map()`` to sample obstacles).
    """
    path = resolve_map_path(map_name)
    with open(path.with_name(path.name + ".yaml"), encoding="utf-8") as handle:
        header = yaml.safe_load(handle)
    has_grid = path.with_name(path.name + ".cfg").exists()
    if not has_grid and "submaporigin" in header:
        return DynamicMap(path.parent, path.name, margin2wall=margin2wall)
    return GridMap(path, margin2wall=margin2wall)


def load_map(map_name: PathLike) -> tuple[np.ndarray, list[float], list[float]]:
    """Return ``(occupancy, mapmin, mapmax)`` for plotting.

    ``occupancy`` has shape ``(ny, nx)`` (rows indexed by y) and is all zeros
    for empty and dynamic maps.
    """
    grid = load_grid_map(map_name)
    nx, ny = grid.mapdim
    occupancy = np.zeros((ny, nx)) if grid.map is None else np.asarray(grid.map, dtype=float)
    return occupancy, grid.mapmin.tolist(), grid.mapmax.tolist()
