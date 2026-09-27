"""Tests for the AJLATT occupancy maps.

The vectorised ray casting is compared against a verbatim port of the original
cell-by-cell implementation (``_Reference``), which it must match exactly.
"""

import numpy as np
import pytest

from env_lib.ajlatt_env import maps
from env_lib.ajlatt_env.maps import DynamicMap, GridMap, bresenham2D, load_grid_map


# ---------------------------------------------------------------------------
# Verbatim port of the original (loop-based) implementation.
# ---------------------------------------------------------------------------
def _round(x):
    return int(x + 0.5) if x >= 0 else int(x - 0.5)


def _ref_bresenham2D(sx, sy, ex, ey):
    sx, sy, ex, ey = _round(sx), _round(sy), _round(ex), _round(ey)
    dx, dy = abs(ex - sx), abs(ey - sy)
    steep = abs(dy) > abs(dx)
    if steep:
        dx, dy = dy, dx
    if dy == 0:
        q = np.zeros((dx + 1, 1))
    else:
        q = np.append(
            0,
            np.greater_equal(
                np.diff(
                    np.mod(np.arange(np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - 1, -dy), dx)
                ),
                0,
            ),
        )
    if steep:
        y = np.arange(sy, ey + 1) if sy <= ey else np.arange(sy, ey - 1, -1)
        x = sx + np.cumsum(q) if sx <= ex else sx - np.cumsum(q)
    else:
        x = np.arange(sx, ex + 1) if sx <= ex else np.arange(sx, ex - 1, -1)
        y = sy + np.cumsum(q) if sy <= ey else sy - np.cumsum(q)
    return np.vstack((x, y)).astype(np.int16)


class _Reference:
    def __init__(self, grid: GridMap):
        self.g = grid
        self.map_linear = None if grid.map is None else grid.map.reshape(-1).astype(np.int8)

    def se2_to_cell(self, pos):
        c = (np.asarray(pos)[:2] - self.g.mapmin) / self.g.mapres - 0.5
        return _round(c[0]), _round(c[1])

    def cell_to_se2(self, cell):
        return (np.array(cell) + 0.5) * self.g.mapres + self.g.mapmin

    def is_collision_ray_cell(self, cell):
        d = self.g.mapdim
        if cell[0] < 0 or cell[1] < 0 or cell[0] >= d[0] or cell[1] >= d[1]:
            return True
        return self.map_linear is not None and self.map_linear[cell[0] + d[0] * cell[1]] == 1

    def is_blocked(self, start, end):
        if self.g.map is None:
            return False
        s, e = self.se2_to_cell(start), self.se2_to_cell(end)
        cells = _ref_bresenham2D(s[0], s[1], e[0], e[1])
        return any(self.is_collision_ray_cell(cells[:, i]) for i in range(cells.shape[1]))

    def get_closest_obstacle(self, odom, ang_res=0.05, fov=np.pi, r_max=3.0):
        odom = np.array(odom)
        ang_grid = np.arange(-0.5 * fov, 0.5 * fov, ang_res)
        closest = (r_max, 0.0)
        start = self.se2_to_cell(odom[:2])
        cs = r_max * np.array([np.cos(ang_grid), np.sin(ang_grid)])
        rot = np.array(
            [[np.cos(odom[-1]), -np.sin(odom[-1])], [np.sin(odom[-1]), np.cos(odom[-1])]]
        )
        end = rot @ cs + odom[:2, np.newaxis]
        ex = np.round((end.T[:, 0] - self.g.mapmin[0]) / self.g.mapres[0] - 0.5)
        ey = np.round((end.T[:, 1] - self.g.mapmin[1]) / self.g.mapres[1] - 0.5)
        for j in range(len(ang_grid)):
            cells = _ref_bresenham2D(start[0], start[1], ex[j], ey[j])
            i = 0
            if self.g.map is None:
                while i < cells.shape[-1]:
                    pt = self.cell_to_se2(cells[:, i])
                    if not self.g.in_bound(pt):
                        break
                    i += 1
                if i < cells.shape[-1]:
                    ro = np.sqrt(np.sum(np.square(pt - odom[:2])))
                    if ro < closest[0]:
                        closest = (ro, ang_grid[j])
            else:
                while i < cells.shape[-1]:
                    if self.is_collision_ray_cell(cells[:, i]):
                        break
                    i += 1
                if i < cells.shape[-1]:
                    ro = np.sqrt(np.sum(np.square(self.cell_to_se2(cells[:, i]) - odom[:2])))
                    if ro < closest[0]:
                        closest = (ro, ang_grid[j])
        return None if closest[0] == r_max else closest


def _random_poses(grid, n, rng, pad=2.0):
    lo, hi = grid.mapmin - pad, grid.mapmax + pad
    xy = rng.uniform(lo, hi, size=(n, 2))
    th = rng.uniform(-np.pi, np.pi, size=(n, 1))
    return np.hstack([xy, th])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_bresenham_matches_reference_exhaustively():
    for sx in range(-3, 4):
        for sy in range(-3, 4):
            for ex in range(-9, 10):
                for ey in range(-9, 10):
                    np.testing.assert_array_equal(
                        bresenham2D(sx, sy, ex, ey), _ref_bresenham2D(sx, sy, ex, ey)
                    )


@pytest.mark.parametrize("name", ["obstacles04", "obstacles05", "obstacles02", "empty"])
def test_closest_obstacle_matches_reference(name):
    grid = load_grid_map(name, margin2wall=1.0)
    ref = _Reference(grid)
    rng = np.random.default_rng(0)
    for fov, r_max in [(np.pi, 3.0), (2 * np.pi, 3.0), (np.pi / 2, 6.0)]:
        for pose in _random_poses(grid, 150, rng):
            expected = ref.get_closest_obstacle(pose, fov=fov, r_max=r_max)
            actual = grid.get_closest_obstacle(pose, fov=fov, r_max=r_max)
            if expected is None:
                assert actual is None
            else:
                assert actual is not None
                assert actual[0] == expected[0]
                assert actual[1] == expected[1]


@pytest.mark.parametrize("name", ["obstacles04", "obstacles05"])
def test_is_blocked_matches_reference(name):
    grid = load_grid_map(name)
    ref = _Reference(grid)
    rng = np.random.default_rng(1)
    starts = _random_poses(grid, 400, rng, pad=0.5)
    ends = _random_poses(grid, 400, rng, pad=0.5)
    for s, e in zip(starts, ends):
        assert grid.is_blocked(s, e) == ref.is_blocked(s, e)


def test_empty_map_is_never_blocked():
    grid = load_grid_map("empty")
    assert grid.map is None
    assert not grid.is_blocked([1.0, 1.0, 0.0], [30.0, 20.0, 0.0])


def test_is_collision_margin():
    occupancy = np.zeros((20, 30))
    occupancy[10, 15] = 1  # cell (cx=15, cy=10)
    grid = GridMap.from_array(occupancy, mapres=1.0, margin2wall=1.0)
    centre = grid.cell_to_se2([15, 10])
    assert grid.is_collision(centre, margin=0.0)
    assert grid.is_collision(centre + [1.0, 0.0], margin=2.0)
    assert not grid.is_collision(centre + [5.0, 5.0], margin=1.0)
    assert grid.is_collision([0.2, 5.0])  # outside the wall margin


def test_available_maps_and_resolution(tmp_path):
    names = maps.available_maps()
    for expected in ["obstacles04", "obstacles05", "empty", "dynamic_map"]:
        assert expected in names
    with pytest.raises(FileNotFoundError):
        maps.resolve_map_path("does_not_exist")

    occupancy = np.zeros((5, 8))
    occupancy[2, 3] = 1
    np.savetxt(tmp_path / "tiny.cfg", occupancy, fmt="%d")
    (tmp_path / "tiny.yaml").write_text(
        "mapdim: [8, 5]\nmapres: [1.0, 1.0]\nmapmin: [0.0, 0.0]\nmapmax: [8.0, 5.0]\n"
    )
    grid = load_grid_map(tmp_path / "tiny.yaml")
    assert grid.mapdim == [8, 5]
    assert grid.is_collision_ray_cell([3, 2])
    assert not grid.is_collision_ray_cell([2, 3])


def test_short_map_file_is_padded_with_walls(tmp_path):
    np.savetxt(tmp_path / "short.cfg", np.zeros((4, 6)), fmt="%d")
    (tmp_path / "short.yaml").write_text(
        "mapdim: [6, 5]\nmapres: [1.0, 1.0]\nmapmin: [0.0, 0.0]\nmapmax: [6.0, 5.0]\n"
    )
    with pytest.warns(UserWarning, match="treated as obstacles"):
        grid = load_grid_map(tmp_path / "short")
    assert grid.map.shape == (5, 6)
    assert grid.map[-1].all()


def test_bundled_maps_have_consistent_dimensions():
    for name in maps.available_maps():
        grid = load_grid_map(name)
        if grid.map is not None:
            assert grid.map.shape == (grid.mapdim[1], grid.mapdim[0]), name


def test_dynamic_map_generation_is_seeded():
    grid = load_grid_map("dynamic_map")
    assert isinstance(grid, DynamicMap)
    a = grid.generate_map(rng=np.random.default_rng(3)).copy()
    b = grid.generate_map(rng=np.random.default_rng(3)).copy()
    c = grid.generate_map(rng=np.random.default_rng(4)).copy()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a[0].all() and a[:, 0].all()
    assert 0.02 < a.mean() < 0.5


def test_is_blocked_batch_matches_scalar():
    grid = load_grid_map("obstacles04")
    rng = np.random.default_rng(5)
    starts = _random_poses(grid, 300, rng, pad=0.5)
    ends = _random_poses(grid, 300, rng, pad=0.5)
    batch = grid.is_blocked_batch(starts, ends)
    scalar = np.array([grid.is_blocked(s, e) for s, e in zip(starts, ends)])
    np.testing.assert_array_equal(batch, scalar)
    assert batch.any() and not batch.all()
