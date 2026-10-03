"""Record environment rollouts and export them as GIF or MP4 files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

__all__ = ["record_episode", "save_animation"]

PathLike = Union[str, Path]


def save_animation(
    frames: Sequence[np.ndarray],
    path: PathLike,
    fps: float = 30.0,
    *,
    loop: int = 0,
) -> Path:
    """Write a sequence of RGB frames to ``path``.

    The format is chosen from the file suffix:

    * ``.gif`` -- written with Pillow (always available, since Pillow is a
      matplotlib dependency).
    * ``.mp4``, ``.webm``, ``.avi``, ``.mov`` -- written with ``imageio`` and
      ``imageio-ffmpeg`` (install with ``pip install "my-tool-box[video]"``).

    Parameters
    ----------
    frames:
        ``(H, W, 3)`` ``uint8`` arrays of identical shape.
    path:
        Output file.
    fps:
        Playback frame rate.
    loop:
        GIF loop count (``0`` loops forever).

    Returns
    -------
    pathlib.Path
        The written file.
    """
    if len(frames) == 0:
        raise ValueError("save_animation() needs at least one frame")
    stack = [np.asarray(frame) for frame in frames]
    shape = stack[0].shape
    if len(shape) != 3 or shape[-1] not in (3, 4):
        raise ValueError(f"frames must have shape (H, W, 3) or (H, W, 4), got {shape}")
    if any(frame.shape != shape for frame in stack):
        raise ValueError("all frames must have the same shape")
    stack = [np.clip(frame, 0, 255).astype(np.uint8, copy=False) for frame in stack]
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".gif":
        from PIL import Image

        images = [Image.fromarray(frame) for frame in stack]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=max(1, int(round(1000.0 / fps))),
            loop=loop,
            optimize=False,
        )
        return path

    if suffix in {".mp4", ".webm", ".avi", ".mov", ".mkv"}:
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg  # noqa: F401  (the MP4 backend of imageio)
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                'Video export requires imageio and imageio-ffmpeg: pip install "my-tool-box[video]"'
            ) from exc
        # Most codecs require even frame dimensions.
        height, width = shape[0] - shape[0] % 2, shape[1] - shape[1] % 2
        with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
            for frame in stack:
                writer.append_data(frame[:height, :width, :3])
        return path

    raise ValueError(f"Unsupported animation format {suffix!r}; use .gif or .mp4")


def record_episode(
    env: Any,
    policy: Callable[[Any], Any] | None = None,
    path: PathLike | None = None,
    *,
    max_steps: int | None = None,
    seed: int | None = None,
    fps: float | None = None,
    render_every: int = 1,
) -> list[np.ndarray]:
    """Roll out one episode and collect the rendered frames.

    Parameters
    ----------
    env:
        A Gymnasium environment created with ``render_mode="rgb_array"``.
    policy:
        Callable mapping an observation to an action. Defaults to uniformly
        random actions from ``env.action_space``.
    path:
        If given, the frames are written with :func:`save_animation`.
    max_steps:
        Optional cap on the number of environment steps.
    seed:
        Seed passed to ``env.reset``.
    fps:
        Playback rate for the saved file; defaults to
        ``env.metadata["render_fps"]`` (or 30).
    render_every:
        Render one frame every ``render_every`` steps.

    Returns
    -------
    list of numpy.ndarray
        The collected RGB frames (the first frame shows the initial state).
    """
    if getattr(env, "render_mode", None) != "rgb_array":
        raise ValueError("record_episode() requires an environment with render_mode='rgb_array'")
    if render_every < 1:
        raise ValueError("render_every must be >= 1")

    act = policy if policy is not None else (lambda _obs: env.action_space.sample())
    observation, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = [env.render()]
    step = 0
    while max_steps is None or step < max_steps:
        observation, _, terminated, truncated, _ = env.step(act(observation))
        step += 1
        if step % render_every == 0:
            frames.append(env.render())
        if np.any(terminated) or np.any(truncated):
            break

    if path is not None:
        rate = fps or env.metadata.get("render_fps", 30) or 30
        save_animation(frames, path, fps=rate)
    return frames
