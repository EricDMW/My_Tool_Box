"""Procedurally generated maps built from a library of obstacle shapes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from scipy import ndimage

from env_lib.ajlatt_env.maps.grid_map import GridMap

__all__ = ["DynamicMap"]

PathLike = Union[str, Path]


class DynamicMap(GridMap):
    """Grid map whose obstacles are re-sampled by :meth:`generate_map`.

    The header (``<map_name>.yaml``) must define ``submaporigin`` (four
    ``(row, col)`` cell anchors, flattened) and ``lib_path`` (a directory of
    ``.npy`` obstacle masks, relative to ``map_dir_path``). Every call to
    :meth:`generate_map` places four distinct obstacles from the library,
    rotated by a random multiple of 18 degrees, at the anchors, and closes the
    border.

    Parameters
    ----------
    map_dir_path:
        Directory containing the header and the obstacle library.
    map_name:
        Header name without extension.
    margin2wall:
        Safety margin used by :meth:`in_bound`.
    map_path:
        Deprecated and ignored (kept for signature compatibility).
    """

    def __init__(
        self,
        map_dir_path: PathLike,
        map_name: str,
        map_path: PathLike | None = None,
        margin2wall: float = 0.5,
    ):
        map_dir = Path(map_dir_path)
        with open(map_dir / f"{map_name}.yaml", encoding="utf-8") as handle:
            header = yaml.safe_load(handle)
        anchors = header.get("submaporigin")
        if anchors is None or len(anchors) < 8:
            raise ValueError(f"Dynamic map {map_name!r} must define 8 'submaporigin' values")
        self.submap_coordinates = [[int(anchors[2 * i]), int(anchors[2 * i + 1])] for i in range(4)]

        library = map_dir / str(header.get("lib_path", "lib_obstacles")).strip()
        files = sorted(library.glob("*.npy"))
        if len(files) < 4:
            raise FileNotFoundError(
                f"Dynamic map {map_name!r} needs at least 4 obstacle masks in {library}"
            )
        self.obstacles = [np.load(path) for path in files]
        self.chosen_idx: np.ndarray | None = None
        self.rot_angs: Sequence[float] | None = None

        nx, ny = (int(v) for v in header["mapdim"])
        self._init_from_header(header, np.zeros((ny, nx)), margin2wall, name=map_name)

    def generate_map(
        self,
        chosen_idx: Sequence[int] | None = None,
        rot_angs: Sequence[float] | None = None,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Sample a new obstacle layout.

        Parameters
        ----------
        chosen_idx:
            Indices of the four library obstacles to place (sampled if ``None``).
        rot_angs:
            Rotation of each obstacle in degrees (sampled if ``None``).
        rng:
            Random generator used for sampling (a fresh one if ``None``).

        Returns
        -------
        numpy.ndarray
            The new occupancy grid, shape ``(ny, nx)``.
        """
        rng = np.random.default_rng() if rng is None else rng
        if chosen_idx is None:
            chosen_idx = rng.choice(len(self.obstacles), 4, replace=False)
        if rot_angs is None:
            rot_angs = [float(rng.choice(np.arange(-10, 10, 1) / 10.0 * 180.0)) for _ in range(4)]

        nx, ny = self.mapdim
        grid = np.zeros((ny, nx))
        for anchor, index, angle in zip(self.submap_coordinates, chosen_idx, rot_angs):
            mask = ndimage.rotate(self.obstacles[int(index)], angle, reshape=True, order=0) > 0.5
            rows, cols = np.nonzero(mask)
            rows = rows - mask.shape[0] // 2 + anchor[0]
            cols = cols - mask.shape[1] // 2 + anchor[1]
            keep = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
            grid[rows[keep], cols[keep]] = 1.0
        grid[0, :] = grid[-1, :] = 1.0
        grid[:, 0] = grid[:, -1] = 1.0

        self.map = grid
        self.map_linear = grid.reshape(-1).astype(np.int8)
        self._occupied = (grid == 1).T.copy()
        self.chosen_idx = np.asarray(chosen_idx)
        self.rot_angs = list(rot_angs)
        return grid
