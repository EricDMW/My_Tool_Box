"""Build AJLATT occupancy maps from YAML specifications.

A specification describes the map size and a list of obstacles in metres::

    map_info:
      name: my_map
      width: 36.0          # metres along x
      height: 36.0         # metres along y
      resolution: 0.2      # cell size in metres
      boundary:
        enabled: true
        thickness: 0.2
        style: rectangle   # rectangle | circle | polygon
        rectangle: {margin: 0.0}
    obstacles:
      - {type: rectangle, center: [10, 10], width: 4, height: 2, angle: 0.3}
      - {type: circle, center: [25, 25], radius: 3, filled: false}
      - {type: polygon, points: [[5, 20], [9, 20], [7, 26]]}
      - {type: line, start: [0, 18], end: [12, 18], width: 0.4}
      - {type: random, num_obstacles: 5, min_size: 0.5, max_size: 1.5, seed: 3}

Supported obstacle types: ``rectangle`` (``center``, ``width``, ``height``,
``angle`` in radians), ``circle`` (``center``, ``radius``), ``triangle``
(``vertices``), ``hexagon`` (``center``, ``radius``, ``angle``), ``polygon``
(``points``), ``line`` (``start``, ``end``, ``width``), ``maze``
(``cell_size``, ``wall_thickness``, ``gaps``) and ``random``
(``num_obstacles``, ``min_size``, ``max_size``, ``seed``). Closed shapes accept
``filled: false`` for a one-cell outline.

A cell is occupied when its centre lies inside a shape. The generated grid is
written row-major with rows indexed by y, which is the layout
:class:`~env_lib.ajlatt_env.maps.GridMap` reads.

The header written next to the grid embeds the specification, so a map can
always be rebuilt from its own ``.yaml`` file.

Command line::

    ajlatt-build-map --create-sample my_map        # writes my_map.spec.yaml (map_info.name: my_map)
    ajlatt-build-map my_map.spec.yaml --png        # writes <map_info.name>.yaml/.cfg/.png
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml
from matplotlib.path import Path as MplPath

__all__ = ["SAMPLE_SPEC", "build_map", "load_spec", "main", "rasterize", "save_map"]

PathLike = Union[str, Path]

SAMPLE_SPEC = """\
# Sample AJLATT map specification. Build it with:
#   ajlatt-build-map sample_map.spec.yaml --png
map_info:
  name: sample_map
  width: 36.2          # metres along x
  height: 36.2         # metres along y
  resolution: 0.2      # metres per cell
  boundary:
    enabled: true
    thickness: 0.2
    style: rectangle   # rectangle | circle | polygon
    rectangle:
      margin: 0.0

obstacles:
  - type: rectangle
    center: [10.0, 10.0]
    width: 6.0
    height: 3.0
    angle: 0.0
  - type: rectangle
    center: [25.0, 20.0]
    width: 4.0
    height: 6.0
    angle: 0.785
  - type: circle
    center: [5.0, 5.0]
    radius: 1.5
  - type: circle
    center: [30.0, 30.0]
    radius: 2.5
    filled: false
  - type: triangle
    vertices: [[7.5, 25.0], [12.5, 25.0], [10.0, 30.0]]
  - type: hexagon
    center: [15.0, 15.0]
    radius: 2.0
    angle: 0.0
  - type: line
    start: [18.0, 2.0]
    end: [18.0, 8.0]
    width: 0.4
  - type: random
    num_obstacles: 3
    min_size: 0.5
    max_size: 1.2
    seed: 42
"""


# ---------------------------------------------------------------------------
# Specification loading
# ---------------------------------------------------------------------------
def load_spec(spec: PathLike | Mapping[str, Any]) -> dict[str, Any]:
    """Load and normalise a specification (path to YAML or mapping).

    Legacy files that only contain a grid header (``mapdim``, ``mapres``,
    ``mapmax``) are converted into an obstacle-free specification.
    """
    name = None
    if not isinstance(spec, Mapping):
        path = Path(spec)
        with open(path, encoding="utf-8") as handle:
            spec = yaml.safe_load(handle) or {}
        name = path.stem
    spec = dict(spec)
    if "map_info" not in spec:
        if "mapmax" not in spec or "mapres" not in spec:
            raise ValueError("specification needs a 'map_info' section")
        mapmin = spec.get("mapmin", [0.0, 0.0])
        spec = {
            "map_info": {
                "name": name or "map",
                "width": float(spec["mapmax"][0]) - float(mapmin[0]),
                "height": float(spec["mapmax"][1]) - float(mapmin[1]),
                "resolution": float(spec["mapres"][0]),
                "boundary": {"enabled": True},
            },
            "obstacles": [],
        }
    info = dict(spec["map_info"])
    info.setdefault("name", name or "map")
    for key in ("width", "height", "resolution"):
        if key not in info:
            raise ValueError(f"map_info.{key} is required")
        info[key] = float(info[key])
        if info[key] <= 0:
            raise ValueError(f"map_info.{key} must be positive")
    spec["map_info"] = info
    spec["obstacles"] = list(spec.get("obstacles") or [])
    return spec


# ---------------------------------------------------------------------------
# Geometry helpers (vectorised over cell centres)
# ---------------------------------------------------------------------------
def _segment_distance(
    x: np.ndarray, y: np.ndarray, a: Sequence[float], b: Sequence[float]
) -> np.ndarray:
    ax, ay = map(float, a)
    bx, by = map(float, b)
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return np.hypot(x - ax, y - ay)
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / length2, 0.0, 1.0)
    return np.hypot(x - (ax + t * dx), y - (ay + t * dy))


def _polygon(
    points: Sequence[Sequence[float]], x: np.ndarray, y: np.ndarray, filled: bool, res: float
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("a polygon needs at least three [x, y] points")
    if filled:
        inside = MplPath(points).contains_points(np.column_stack([x.ravel(), y.ravel()]))
        return inside.reshape(x.shape)
    return _outline(points, x, y, res)


def _outline(points: np.ndarray, x: np.ndarray, y: np.ndarray, width: float) -> np.ndarray:
    mask = np.zeros(x.shape, dtype=bool)
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        mask |= _segment_distance(x, y, a, b) <= width / 2
    return mask


def _regular_polygon(center, radius: float, sides: int, angle: float) -> np.ndarray:
    theta = angle + 2 * np.pi * np.arange(sides) / sides
    return np.column_stack([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)])


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------
def rasterize(spec: PathLike | Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterise a specification.

    Returns
    -------
    occupancy:
        ``int8`` array of shape ``(ny, nx)`` (rows indexed by y), 1 = occupied.
    header:
        The grid header to be written as ``<name>.yaml``.
    """
    spec = load_spec(spec)
    info = spec["map_info"]
    res = info["resolution"]
    nx = int(round(info["width"] / res))
    ny = int(round(info["height"] / res))
    if nx < 3 or ny < 3:
        raise ValueError("the map must be at least 3 cells wide and high")
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    x, y = np.meshgrid(xs, ys)  # shape (ny, nx)
    occupancy = np.zeros((ny, nx), dtype=bool)

    boundary = dict(info.get("boundary") or {})
    if boundary.get("enabled", True):
        occupancy |= _boundary(boundary, x, y, nx, ny, info, res)

    for index, obstacle in enumerate(spec["obstacles"]):
        try:
            occupancy |= _obstacle(dict(obstacle), x, y, nx, ny, info, res)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid obstacle #{index} ({obstacle!r}): {exc}") from exc

    header = {
        "datatype": "t",
        "mapdim": [nx, ny],
        "mapmax": [round(nx * res, 10), round(ny * res, 10)],
        "mapmin": [0.0, 0.0],
        "mappath": f"{info['name']}.cfg",
        "mapres": [res, res],
        "origin": [round(nx * res / 2, 10), round(ny * res / 2, 10)],
        "origincells": [nx // 2, ny // 2],
        "storage": "colmajor",
    }
    return occupancy.astype(np.int8), header


def _boundary(boundary, x, y, nx, ny, info, res) -> np.ndarray:
    style = boundary.get("style", "rectangle")
    thickness = max(1, int(round(float(boundary.get("thickness", res)) / res)))
    mask = np.zeros((ny, nx), dtype=bool)
    if style == "rectangle":
        margin = int(round(float((boundary.get("rectangle") or {}).get("margin", 0.0)) / res))
        lo_x, hi_x, lo_y, hi_y = margin, nx - margin, margin, ny - margin
        mask[lo_y : lo_y + thickness, lo_x:hi_x] = True
        mask[hi_y - thickness : hi_y, lo_x:hi_x] = True
        mask[lo_y:hi_y, lo_x : lo_x + thickness] = True
        mask[lo_y:hi_y, hi_x - thickness : hi_x] = True
        return mask
    if style == "circle":
        circle = boundary.get("circle") or {}
        cx, cy = circle.get("center", [info["width"] / 2, info["height"] / 2])
        radius = float(circle.get("radius", min(info["width"], info["height"]) / 2))
        return np.hypot(x - cx, y - cy) > radius - thickness * res
    if style == "polygon":
        points = np.asarray((boundary.get("polygon") or {})["points"], dtype=float)
        inside = (
            MplPath(points)
            .contains_points(np.column_stack([x.ravel(), y.ravel()]))
            .reshape(x.shape)
        )
        return ~inside | _outline(points, x, y, thickness * res)
    raise ValueError(f"unknown boundary style {style!r}")


def _obstacle(obstacle, x, y, nx, ny, info, res) -> np.ndarray:
    kind = obstacle.get("type")
    filled = bool(obstacle.get("filled", True))
    if kind == "rectangle":
        cx, cy = map(float, obstacle["center"])
        half_w, half_h = float(obstacle["width"]) / 2, float(obstacle["height"]) / 2
        angle = float(obstacle.get("angle", 0.0))
        c, s = math.cos(angle), math.sin(angle)
        u = (x - cx) * c + (y - cy) * s
        v = -(x - cx) * s + (y - cy) * c
        inside = (np.abs(u) <= half_w) & (np.abs(v) <= half_h)
        if filled:
            return inside
        return inside & ~((np.abs(u) <= half_w - res) & (np.abs(v) <= half_h - res))
    if kind == "circle":
        cx, cy = map(float, obstacle["center"])
        radius = float(obstacle["radius"])
        d = np.hypot(x - cx, y - cy)
        return d <= radius if filled else (d <= radius) & (d > radius - res)
    if kind == "triangle":
        vertices = obstacle["vertices"]
        if len(vertices) != 3:
            raise ValueError("a triangle needs exactly three vertices")
        return _polygon(vertices, x, y, filled, res)
    if kind == "hexagon":
        points = _regular_polygon(
            obstacle["center"], float(obstacle["radius"]), 6, float(obstacle.get("angle", 0.0))
        )
        return _polygon(points, x, y, filled, res)
    if kind == "polygon":
        return _polygon(obstacle["points"], x, y, filled, res)
    if kind == "line":
        width = max(float(obstacle.get("width", res)), res)
        return _segment_distance(x, y, obstacle["start"], obstacle["end"]) <= width / 2
    if kind == "maze":
        return _maze(obstacle, nx, ny, res)
    if kind == "random":
        return _random_blocks(obstacle, nx, ny, info, res)
    raise ValueError(f"unknown obstacle type {kind!r}")


def _maze(pattern, nx, ny, res) -> np.ndarray:
    mask = np.zeros((ny, nx), dtype=bool)
    cell = max(1, int(round(float(pattern.get("cell_size", 5.0)) / res)))
    wall = max(1, int(round(float(pattern.get("wall_thickness", 0.5)) / res)))
    gaps = pattern.get("gaps") or []
    horizontal_gaps = {int(g["position"]) for g in gaps if g.get("type") == "horizontal"}
    vertical_gaps = {int(g["position"]) for g in gaps if g.get("type") == "vertical"}
    for row in range(cell, ny - cell, cell):
        if row not in horizontal_gaps:
            mask[row : row + wall, cell : nx - cell] = True
    for col in range(cell, nx - cell, cell):
        if col not in vertical_gaps:
            mask[cell : ny - cell, col : col + wall] = True
    return mask


def _random_blocks(pattern, nx, ny, info, res) -> np.ndarray:
    mask = np.zeros((ny, nx), dtype=bool)
    rng = np.random.default_rng(pattern.get("seed"))
    count = int(pattern.get("num_obstacles", 10))
    low, high = float(pattern.get("min_size", 1.0)), float(pattern.get("max_size", 3.0))
    for _ in range(count):
        cx = rng.uniform(2.0, info["width"] - 2.0) / res
        cy = rng.uniform(2.0, info["height"] - 2.0) / res
        half = int(rng.uniform(low, high) / res)
        cx, cy = int(cx), int(cy)
        mask[max(0, cy - half) : min(ny, cy + half), max(0, cx - half) : min(nx, cx + half)] = True
    return mask


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_map(
    occupancy: np.ndarray,
    header: Mapping[str, Any],
    output_dir: PathLike,
    name: str | None = None,
    spec: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``<name>.yaml`` and ``<name>.cfg``; return both paths.

    When ``spec`` is given it is embedded in the header file (``map_info`` and
    ``obstacles`` keys), so the map can be rebuilt from its own header.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = name or Path(str(header.get("mappath", "map.cfg"))).stem
    content: dict[str, Any] = dict(header, mappath=f"{name}.cfg")
    if spec is not None:
        content["map_info"] = dict(spec["map_info"], name=name)
        content["obstacles"] = list(spec.get("obstacles", []))
    yaml_path = output_dir / f"{name}.yaml"
    cfg_path = output_dir / f"{name}.cfg"
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Generated by ajlatt-build-map; grid header followed by its specification.\n"
        )
        yaml.safe_dump(content, handle, default_flow_style=None, sort_keys=False)
    np.savetxt(cfg_path, np.asarray(occupancy, dtype=np.int8), fmt="%d")
    return yaml_path, cfg_path


def build_map(
    spec: PathLike | Mapping[str, Any],
    output_dir: PathLike | None = None,
    *,
    name: str | None = None,
    png: bool = False,
) -> tuple[Path, Path]:
    """Rasterise ``spec`` and write the map files.

    Parameters
    ----------
    spec:
        Path to a YAML specification or an equivalent mapping.
    output_dir:
        Destination directory (defaults to the directory of ``spec``, or the
        working directory for mappings).
    name:
        Output name (defaults to ``map_info.name``).
    png:
        Also write ``<name>.png`` with a rendering of the map.

    Returns
    -------
    tuple of pathlib.Path
        Paths of the written header and grid files.
    """
    loaded = load_spec(spec)
    occupancy, header = rasterize(loaded)
    if output_dir is None:
        output_dir = Path(spec).parent if not isinstance(spec, Mapping) else Path.cwd()
    name = name or loaded["map_info"]["name"]
    yaml_path, cfg_path = save_map(occupancy, header, output_dir, name, spec=loaded)
    if png:
        from env_lib.ajlatt_env.maps.plotting import plot_map

        plot_map(yaml_path.with_suffix(""), save=yaml_path.with_suffix(".png"))
    return yaml_path, cfg_path


def main(argv: Iterable[str] | None = None) -> int:
    """Command-line entry point (``ajlatt-build-map``)."""
    parser = argparse.ArgumentParser(
        prog="ajlatt-build-map", description="Build AJLATT occupancy maps from YAML specifications."
    )
    parser.add_argument("spec", nargs="?", help="YAML specification to build")
    parser.add_argument("-o", "--output-dir", help="directory for the generated files")
    parser.add_argument("-n", "--name", help="output map name (default: map_info.name)")
    parser.add_argument("--png", action="store_true", help="also write a PNG preview")
    parser.add_argument(
        "--create-sample", metavar="PATH", help="write a sample specification and exit"
    )
    args = parser.parse_args(None if argv is None else list(argv))

    if args.create_sample:
        path = Path(args.create_sample)
        if not path.name.endswith(".yaml"):
            path = path.with_name(path.name + ".spec.yaml")
        name = path.name.split(".")[0]
        path.write_text(SAMPLE_SPEC.replace("name: sample_map", f"name: {name}"), encoding="utf-8")
        print(f"wrote sample specification to {path}")
        return 0
    if not args.spec:
        parser.error("a specification file is required (or use --create-sample)")
    yaml_path, cfg_path = build_map(args.spec, args.output_dir, name=args.name, png=args.png)
    if yaml_path.resolve() == Path(args.spec).resolve():
        print(f"note: {yaml_path} now holds the grid header followed by the original specification")
    occupancy = np.loadtxt(cfg_path)
    print(
        f"wrote {yaml_path} and {cfg_path}: {occupancy.shape[1]} x {occupancy.shape[0]} cells, "
        f"{100 * occupancy.mean():.1f}% occupied"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
