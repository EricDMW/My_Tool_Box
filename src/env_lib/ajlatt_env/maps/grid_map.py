"""Occupancy grid maps with vectorised ray casting.

Coordinate conventions
----------------------
* A map covers the rectangle ``[mapmin, mapmax]`` with cells of size ``mapres``.
  ``mapdim = (nx, ny)`` is the number of cells along x and y.
* Cell ``(cx, cy)`` has its centre at ``(cell + 0.5) * mapres + mapmin``.
* The occupancy file (``<name>.cfg``) stores the grid row-major with **rows
  indexed by y** and columns by x, i.e. ``occupancy[cy, cx]``; this is also what
  ``imshow(occupancy, origin="lower")`` displays.

All queries are exact re-implementations of the original cell-by-cell loops:
ray casting uses a batched form of the same Bresenham variant, so results
(including the cell a ray stops at) are identical, only computed with NumPy
array operations instead of Python loops.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np
import yaml

__all__ = [
    "GridMap",
    "bresenham2D",
    "bresenham_batch",
    "cell_to_se2",
    "cell_to_se2_batch",
    "coord_change2g",
    "round_half_away",
    "se2_to_cell",
    "se2_to_cell_batch",
]

PathLike = Union[str, Path]

DEFAULT_SENSOR_RANGE = 3.0
DEFAULT_FOV = np.pi


# ---------------------------------------------------------------------------
# Scalar and batched coordinate helpers
# ---------------------------------------------------------------------------
def round_half_away(x):
    """Round half away from zero (the rounding used by the original grid code)."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def _round_scalar(x: float) -> int:
    return int(x + 0.5) if x >= 0 else int(x - 0.5)


def se2_to_cell(pos, mapmin, mapres) -> tuple[int, int]:
    """Cell index ``(cx, cy)`` containing the planar position ``pos[:2]``."""
    cell_idx = (np.asarray(pos, dtype=float)[:2] - mapmin) / mapres - 0.5
    return _round_scalar(cell_idx[0]), _round_scalar(cell_idx[1])


def cell_to_se2(cell_idx, mapmin, mapres) -> np.ndarray:
    """Centre ``(x, y)`` of a cell."""
    return (np.asarray(cell_idx) + 0.5) * mapres + mapmin


def se2_to_cell_batch(pos, mapmin, mapres):
    """Batched conversion with NumPy's round-half-to-even, as in the original.

    Parameters
    ----------
    pos:
        Array of shape ``(batch, 2 or 3)``.

    Returns
    -------
    tuple of numpy.ndarray
        Cell x and y indices (as floats), each of shape ``(batch,)``.
    """
    pos = np.asarray(pos, dtype=float)
    return (
        np.round((pos[:, 0] - mapmin[0]) / mapres[0] - 0.5),
        np.round((pos[:, 1] - mapmin[1]) / mapres[1] - 0.5),
    )


def cell_to_se2_batch(cell_idx, mapmin, mapres):
    """Batched inverse of :func:`se2_to_cell_batch` for an array of shape ``(batch, 2)``."""
    cell_idx = np.asarray(cell_idx, dtype=float)
    return (
        (cell_idx[:, 0] + 0.5) * mapres[0] + mapmin[0],
        (cell_idx[:, 1] + 0.5) * mapres[1] + mapmin[1],
    )


def coord_change2g(vec, ang: float) -> np.ndarray:
    """Rotate ``vec`` (shape ``(2,)`` or ``(2, n)``) from a body frame at angle ``ang``."""
    vec = np.asarray(vec, dtype=float)
    if len(vec) != 2:
        raise ValueError("coord_change2g expects a vector with leading dimension 2")
    return np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ vec


# ---------------------------------------------------------------------------
# Bresenham ray tracing
# ---------------------------------------------------------------------------
def bresenham2D(sx, sy, ex, ey) -> np.ndarray:
    """Cells visited by a ray from ``(sx, sy)`` to ``(ex, ey)`` (both inclusive).

    Returns
    -------
    numpy.ndarray
        ``int16`` array of shape ``(2, n_cells)`` with x indices in row 0 and y
        indices in row 1.
    """
    xs, ys, valid = bresenham_batch(sx, sy, np.atleast_1d(ex), np.atleast_1d(ey))
    n = int(valid[0].sum())
    return np.vstack((xs[0, :n], ys[0, :n])).astype(np.int16)


def bresenham_batch(sx, sy, ex, ey):
    """Batched Bresenham rays.

    This is an exact, vectorised re-implementation of the error-accumulation
    variant used by :func:`bresenham2D` (the major axis advances every step, the
    minor axis whenever ``(floor(a/2) - k*b) mod a`` wraps around).

    Parameters
    ----------
    sx, sy:
        Start cells: scalars (shared start) or arrays of shape ``(n_rays,)``
        (rounded half away from zero).
    ex, ey:
        End cells, arrays of shape ``(n_rays,)`` (rounded half away from zero).

    Returns
    -------
    xs, ys:
        ``int64`` arrays of shape ``(n_rays, max_len)``.
    valid:
        Boolean mask of the same shape; ray ``j`` has ``valid[j].sum()`` cells.
    """
    ex = round_half_away(np.asarray(ex, dtype=float)).astype(np.int64)
    ey = round_half_away(np.asarray(ey, dtype=float)).astype(np.int64)
    sx = np.broadcast_to(round_half_away(sx).astype(np.int64), ex.shape)
    sy = np.broadcast_to(round_half_away(sy).astype(np.int64), ey.shape)

    dx = np.abs(ex - sx)
    dy = np.abs(ey - sy)
    steep = dy > dx
    major = np.where(steep, dy, dx)
    minor = np.where(steep, dx, dy)

    length = int(major.max()) + 1 if major.size else 1
    k = np.arange(length)[None, :]
    valid = k <= major[:, None]

    # Minor-axis increments: q_0 = 0, q_k = [e_k - e_{k-1} >= 0], e_k = (h - k*b) mod a.
    safe_major = np.maximum(major, 1)[:, None]
    half = (major // 2)[:, None]
    err = np.mod(half - k * minor[:, None], safe_major)
    steps = np.zeros_like(err)
    steps[:, 1:] = np.diff(err, axis=1) >= 0
    steps[minor == 0] = 0
    minor_offset = np.cumsum(steps, axis=1)

    sign_x = np.where(ex >= sx, 1, -1)[:, None]
    sign_y = np.where(ey >= sy, 1, -1)[:, None]
    steep_col = steep[:, None]
    x0, y0 = sx[:, None], sy[:, None]
    xs = np.where(steep_col, x0 + sign_x * minor_offset, x0 + sign_x * k)
    ys = np.where(steep_col, y0 + sign_y * k, y0 + sign_y * minor_offset)
    return xs, ys, valid


# ---------------------------------------------------------------------------
# Grid map
# ---------------------------------------------------------------------------
class GridMap:
    """Static occupancy grid map.

    Parameters
    ----------
    map_path:
        Path of the map **without extension**; ``<map_path>.yaml`` holds the
        header and ``<map_path>.cfg`` the occupancy grid. An empty (zero-byte)
        grid file denotes an obstacle-free map in which only the boundary
        blocks rays.
    margin2wall:
        Safety margin (metres) used by :meth:`in_bound` and :meth:`is_collision`.

    Attributes
    ----------
    map:
        Occupancy grid of shape ``(ny, nx)`` (``None`` for empty maps).
    mapdim, mapres, mapmin, mapmax, origin:
        Map header values.
    """

    def __init__(self, map_path: PathLike, margin2wall: float = 0.5):
        map_path = Path(map_path)
        header_path = map_path.with_name(map_path.name + ".yaml")
        with open(header_path, encoding="utf-8") as handle:
            header = yaml.safe_load(handle)
        grid_path = map_path.with_name(map_path.name + ".cfg")
        occupancy = None  # obstacle-free map: only the boundary blocks rays
        if grid_path.exists() and grid_path.stat().st_size > 0:
            occupancy = np.loadtxt(grid_path)
        elif not grid_path.exists() and "empty" not in map_path.name:
            raise FileNotFoundError(f"Occupancy grid {grid_path} not found")
        self._init_from_header(header, occupancy, margin2wall, name=map_path.name)

    @classmethod
    def from_array(
        cls,
        occupancy: np.ndarray | None,
        *,
        mapmin: Sequence[float] = (0.0, 0.0),
        mapres: float | Sequence[float] = 0.4,
        margin2wall: float = 0.5,
        name: str = "custom",
    ) -> GridMap:
        """Build a map from an occupancy array of shape ``(ny, nx)`` (rows = y).

        Pass ``occupancy=None`` together with a header-compatible shape via
        :meth:`from_header` for obstacle-free maps.
        """
        occupancy = np.asarray(occupancy, dtype=float)
        if occupancy.ndim != 2:
            raise ValueError("occupancy must be a 2-D array of shape (ny, nx)")
        res = np.broadcast_to(np.asarray(mapres, dtype=float), (2,)).copy()
        ny, nx = occupancy.shape
        mapmin = np.asarray(mapmin, dtype=float)
        header = {
            "mapdim": [nx, ny],
            "mapres": res.tolist(),
            "mapmin": mapmin.tolist(),
            "mapmax": (mapmin + res * np.array([nx, ny])).tolist(),
            "origin": (mapmin + res * np.array([nx, ny]) / 2).tolist(),
        }
        obj = cls.__new__(cls)
        obj._init_from_header(header, occupancy, margin2wall, name=name)
        return obj

    def _init_from_header(self, header, occupancy, margin2wall, name) -> None:
        self.name = name
        self.mapdim = [int(v) for v in header["mapdim"]]
        self.mapres = np.array(header["mapres"], dtype=float)
        self.mapmin = np.array(header["mapmin"], dtype=float)
        self.mapmax = np.array(header["mapmax"], dtype=float)
        self.origin = header.get("origin")
        self.margin2wall = float(margin2wall)
        nx, ny = self.mapdim
        if occupancy is None:
            self.map = None
            self.map_linear = None
            self._occupied = np.zeros((nx, ny), dtype=bool)
        else:
            flat = np.asarray(occupancy, dtype=float).reshape(-1)
            if flat.size != nx * ny:
                if flat.size < nx * ny:
                    warnings.warn(
                        f"Map {name!r} has {flat.size} cells but its header declares "
                        f"{nx}x{ny}; the missing cells are treated as obstacles.",
                        stacklevel=3,
                    )
                    flat = np.concatenate([flat, np.ones(nx * ny - flat.size)])
                else:
                    raise ValueError(
                        f"Map {name!r} has {flat.size} cells, more than the {nx}x{ny} "
                        "declared in its header"
                    )
            self.map = flat.reshape(ny, nx)
            self.map_linear = flat.astype(np.int8)
            # Same indexing as the original: idx = cx + nx * cy.
            self._occupied = (self.map == 1).T.copy()

    # -- basic conversions --------------------------------------------------
    def se2_to_cell(self, pos) -> tuple[int, int]:
        """Cell containing ``pos[:2]``."""
        return se2_to_cell(pos, self.mapmin, self.mapres)

    def cell_to_se2(self, cell_idx) -> np.ndarray:
        """Centre of ``cell_idx``."""
        return cell_to_se2(cell_idx, self.mapmin, self.mapres)

    def in_bound(self, pos) -> bool:
        """``True`` if ``pos`` is at least ``margin2wall`` inside the map boundary."""
        return not (
            (pos[0] < self.mapmin[0] + self.margin2wall)
            or (pos[0] > self.mapmax[0] - self.margin2wall)
            or (pos[1] < self.mapmin[1] + self.margin2wall)
            or (pos[1] > self.mapmax[1] - self.margin2wall)
        )

    def in_bound_cell(self, cell) -> bool:
        """``True`` if ``cell`` lies inside the grid."""
        return 0 <= cell[0] < self.mapdim[0] and 0 <= cell[1] < self.mapdim[1]

    def _cells_blocked(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorised ``is_collision_ray_cell`` for arrays of cell indices."""
        nx, ny = self.mapdim
        inside = (xs >= 0) & (ys >= 0) & (xs < nx) & (ys < ny)
        blocked = ~inside
        if self.map is not None:
            blocked[inside] = self._occupied[xs[inside], ys[inside]]
        return blocked

    def is_collision_ray_cell(self, cell) -> bool:
        """``True`` if ``cell`` is outside the grid or occupied."""
        return bool(self._cells_blocked(np.asarray([cell[0]]), np.asarray([cell[1]]))[0])

    def is_collision(self, pos, margin: float | None = None) -> bool:
        """``True`` if ``pos`` is out of bounds or within ``margin`` of an occupied cell."""
        if not self.in_bound(pos):
            return True
        if self.map is None:
            return False
        cell = np.minimum([self.mapdim[0] - 1, self.mapdim[1] - 1], self.se2_to_cell(pos))
        if margin is None:
            margin = self.margin2wall
        if margin == 0.0:
            return self.is_collision_ray_cell(cell)
        n = np.ceil(margin / self.mapres).astype(np.int64)
        xs = np.clip(cell[0] + np.arange(-n[1], n[1]), 0, self.mapdim[0] - 1)
        ys = np.clip(cell[1] + np.arange(-n[0], n[0]), 0, self.mapdim[1] - 1)
        return bool(self._occupied[np.ix_(xs, ys)].any())

    # -- ray casting --------------------------------------------------------
    def is_blocked(self, start_pos, end_pos) -> bool:
        """``True`` if the straight line between two positions crosses an obstacle."""
        if self.map is None:
            return False
        sx, sy = self.se2_to_cell(start_pos)
        ex, ey = self.se2_to_cell(end_pos)
        xs, ys, valid = bresenham_batch(sx, sy, np.array([ex]), np.array([ey]))
        return bool(np.any(self._cells_blocked(xs[valid], ys[valid])))

    def is_blocked_batch(self, start_pos, end_pos) -> np.ndarray:
        """Vectorised :meth:`is_blocked` for arrays of shape ``(n, 2 or 3)``."""
        start_pos = np.atleast_2d(np.asarray(start_pos, dtype=float))
        end_pos = np.atleast_2d(np.asarray(end_pos, dtype=float))
        if self.map is None or len(start_pos) == 0:
            return np.zeros(len(start_pos), dtype=bool)
        start = round_half_away((start_pos[:, :2] - self.mapmin) / self.mapres - 0.5)
        end = round_half_away((end_pos[:, :2] - self.mapmin) / self.mapres - 0.5)
        xs, ys, valid = bresenham_batch(start[:, 0], start[:, 1], end[:, 0], end[:, 1])
        return np.any(self._cells_blocked(xs, ys) & valid, axis=1)

    def _cast(self, odom, angles: np.ndarray, r_max: float):
        """Cast rays at ``angles`` (body frame); return first-hit distances and cells.

        Rays that hit nothing get ``inf`` distance.
        """
        odom = np.asarray(odom, dtype=float)
        sx, sy = self.se2_to_cell(odom[:2])
        body = r_max * np.array([np.cos(angles), np.sin(angles)])
        end = coord_change2g(body, odom[-1]) + odom[:2, np.newaxis]
        ex, ey = se2_to_cell_batch(end.T, self.mapmin, self.mapres)
        xs, ys, valid = bresenham_batch(sx, sy, ex, ey)

        if self.map is None:
            cx = (xs + 0.5) * self.mapres[0] + self.mapmin[0]
            cy = (ys + 0.5) * self.mapres[1] + self.mapmin[1]
            hit = (
                (cx < self.mapmin[0] + self.margin2wall)
                | (cx > self.mapmax[0] - self.margin2wall)
                | (cy < self.mapmin[1] + self.margin2wall)
                | (cy > self.mapmax[1] - self.margin2wall)
            )
        else:
            hit = self._cells_blocked(xs, ys)
        hit &= valid

        has_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        rows = np.arange(len(first))
        hit_cells = np.stack([xs[rows, first], ys[rows, first]], axis=1)
        points = (hit_cells + 0.5) * self.mapres + self.mapmin
        dist = np.sqrt(np.sum(np.square(points - odom[:2]), axis=1))
        dist[~has_hit] = np.inf
        return dist, points

    def get_closest_obstacle(
        self,
        odom,
        ang_res: float = 0.05,
        fov: float = DEFAULT_FOV,
        r_max: float = DEFAULT_SENSOR_RANGE,
        return_pt: bool = False,
    ):
        """Range and bearing of the closest obstacle cell within a field of view.

        Parameters
        ----------
        odom:
            Pose ``(x, y, theta)``.
        ang_res:
            Angular resolution of the ray fan (radians).
        fov:
            Field of view (radians), centred on the heading.
        r_max:
            Maximum ray length.
        return_pt:
            Return the obstacle cell centre instead of ``(range, bearing)``.

        Returns
        -------
        tuple or numpy.ndarray or None
            ``(range, bearing)`` of the closest hit, the hit point if
            ``return_pt`` is set, or ``None`` if no ray hits within ``r_max``.
        """
        angles = np.arange(-0.5 * fov, 0.5 * fov, ang_res)
        if angles.size == 0:
            return None
        dist, points = self._cast(odom, angles, r_max)
        best = int(np.argmin(dist))
        if not dist[best] < r_max:
            return None
        if return_pt:
            return points[best]
        return float(dist[best]), float(angles[best])

    def get_front_obstacle(self, odom, r_max: float = DEFAULT_SENSOR_RANGE, **kwargs):
        """Range to the first obstacle straight ahead, as ``(range, 0.0)`` or ``None``."""
        dist, _ = self._cast(odom, np.zeros(1), r_max)
        if not np.isfinite(dist[0]):
            return None
        return float(dist[0]), 0.0

    def __repr__(self) -> str:
        kind = "empty" if self.map is None else f"{int(self.map.sum())} occupied cells"
        return (
            f"{type(self).__name__}(name={self.name!r}, mapdim={self.mapdim}, "
            f"mapres={self.mapres.tolist()}, {kind})"
        )
