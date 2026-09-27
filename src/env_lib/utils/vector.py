"""Base class for natively batched (vectorised) environments.

:class:`gymnasium.vector.SyncVectorEnv` steps ``num_envs`` Python environment
objects one after another, so its cost grows linearly with ``num_envs`` and is
dominated by per-call Python overhead for small models. A
:class:`BatchedVectorEnv` instead keeps the state of all copies in arrays with
a leading batch dimension and advances them with a single set of array
operations, which is typically one to two orders of magnitude faster.

Subclasses implement three hooks:

* :meth:`BatchedVectorEnv._reset_envs` re-initialises the copies selected by a
  boolean mask, drawing randomness from ``self.np_random``;
* :meth:`BatchedVectorEnv._step_envs` advances all copies by one step and
  returns rewards, termination and truncation flags and batched info arrays;
* :meth:`BatchedVectorEnv._observe` returns the batched observation.

The base class implements the Gymnasium vector API on top of them: seeding,
action validation, the three autoreset modes (``"next_step"``, the Gymnasium
default, ``"same_step"`` and ``"disabled"``) and the info-mask convention
(every info key ``k`` comes with a boolean mask ``"_k"``).

Environments created with :func:`env_lib.make_vec` use a native batched
implementation whenever the environment provides one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from env_lib.errors import ResetNeededError

try:  # Gymnasium >= 1.1
    from gymnasium.vector import AutoresetMode
except ImportError:  # pragma: no cover - Gymnasium 1.0
    AutoresetMode = None

__all__ = ["AUTORESET_MODES", "BatchedVectorEnv"]

#: Supported values of the ``autoreset_mode`` argument.
AUTORESET_MODES: tuple[str, ...] = ("next_step", "same_step", "disabled")


class BatchedVectorEnv(VectorEnv):
    """Gymnasium vector environment whose copies are simulated as one batch.

    Parameters
    ----------
    num_envs:
        Number of environment copies ``B``.
    single_observation_space, single_action_space:
        Spaces of one copy. The batched spaces are built with
        :func:`gymnasium.vector.utils.batch_space`.
    autoreset_mode:
        ``"next_step"`` (default, as in Gymnasium): a copy that terminated or
        truncated is reset on the *next* call to :meth:`step`, which ignores
        its action and returns its first observation with zero reward.
        ``"same_step"``: the copy is reset within the same call; the final
        observation is returned in ``infos["final_obs"]`` and its info in
        ``infos["final_info"]``. ``"disabled"``: no automatic reset; call
        ``reset(options={"reset_mask": mask})``.
    render_mode:
        Stored for API compatibility; subclasses that render override
        :meth:`render`.

    Notes
    -----
    All copies share the generator ``self.np_random``: ``reset(seed=s)``
    makes a batch reproducible, but copy ``i`` does not reproduce a single
    environment seeded with ``s + i``.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        num_envs: int,
        single_observation_space: spaces.Space,
        single_action_space: spaces.Space,
        *,
        autoreset_mode: str = "next_step",
        render_mode: str | None = None,
    ) -> None:
        if isinstance(num_envs, bool) or int(num_envs) != num_envs or num_envs < 1:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        # Accept "next_step", "NextStep" and gymnasium.vector.AutoresetMode members.
        raw = str(getattr(autoreset_mode, "value", autoreset_mode))
        mode = {name.replace("_", ""): name for name in AUTORESET_MODES}.get(
            raw.replace("_", "").lower(), raw
        )
        if mode not in AUTORESET_MODES:
            raise ValueError(
                f"autoreset_mode must be one of {AUTORESET_MODES}, got {autoreset_mode!r}"
            )
        self.num_envs = int(num_envs)
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.observation_space = batch_space(single_observation_space, self.num_envs)
        self.action_space = batch_space(single_action_space, self.num_envs)
        self.render_mode = render_mode
        self.autoreset_mode = mode
        self.metadata = dict(type(self).metadata)
        if AutoresetMode is not None:
            self.metadata["autoreset_mode"] = AutoresetMode[mode.upper()]
        self._pending_reset = np.zeros(self.num_envs, dtype=bool)
        self._needs_reset = True

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _reset_envs(self, mask: np.ndarray, options: dict[str, Any] | None) -> None:
        """Re-initialise the copies where ``mask`` is true.

        Parameters
        ----------
        mask:
            Boolean array of shape ``(num_envs,)``.
        options:
            The ``options`` passed to :meth:`reset` (``None`` for automatic
            resets).
        """
        raise NotImplementedError

    def _step_envs(
        self, actions: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Advance the copies by one step.

        Parameters
        ----------
        actions:
            Validated batched action.
        active:
            Boolean mask of the copies whose results are used. Inactive copies
            are reset right afterwards, so implementations may advance them
            anyway (for simplicity) or skip them.

        Returns
        -------
        tuple
            ``(rewards, terminated, truncated, infos)`` with arrays of shape
            ``(num_envs,)`` and a dictionary of arrays with a leading
            ``num_envs`` dimension.
        """
        raise NotImplementedError

    def _observe(self) -> np.ndarray:
        """Batched observation of shape ``(num_envs, *single_observation_space.shape)``."""
        raise NotImplementedError

    def _reset_infos(self, mask: np.ndarray) -> dict[str, Any]:
        """Batched info returned by :meth:`reset` (empty by default)."""
        return {}

    # ------------------------------------------------------------------
    # Gymnasium vector API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset all copies (or those in ``options["reset_mask"]``).

        Returns
        -------
        tuple
            ``(observations, infos)``.
        """
        super().reset(seed=seed)
        mask = np.ones(self.num_envs, dtype=bool)
        if options is not None and "reset_mask" in options:
            mask = np.asarray(options["reset_mask"], dtype=bool)
            if mask.shape != (self.num_envs,):
                raise ValueError(f"reset_mask must have shape ({self.num_envs},), got {mask.shape}")
            if self._needs_reset and not mask.all():
                raise ResetNeededError("The first reset() must reset every sub-environment.")
            options = {key: value for key, value in options.items() if key != "reset_mask"}
        self._reset_envs(mask, options)
        self._pending_reset[mask] = False
        self._needs_reset = False
        return self._observe(), self._with_masks(self._reset_infos(mask))

    def step(
        self, actions: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Advance every copy by one step.

        Returns
        -------
        tuple
            ``(observations, rewards, terminations, truncations, infos)``.
        """
        if self._needs_reset:
            raise ResetNeededError("Call reset() before step().")
        actions = self._validate_actions(actions)
        resetting = self._pending_reset.copy() if self.autoreset_mode == "next_step" else None
        active = ~resetting if resetting is not None else np.ones(self.num_envs, dtype=bool)

        rewards, terminated, truncated, infos = self._step_envs(actions, active)
        rewards = np.array(rewards, dtype=np.float64, copy=True).reshape(self.num_envs)
        terminated = np.array(terminated, dtype=bool, copy=True).reshape(self.num_envs)
        truncated = np.array(truncated, dtype=bool, copy=True).reshape(self.num_envs)
        done = terminated | truncated

        if self.autoreset_mode == "next_step":
            if resetting.any():
                self._reset_envs(resetting, None)
                rewards[resetting] = 0.0
                terminated[resetting] = False
                truncated[resetting] = False
                done[resetting] = False
            self._pending_reset = done
            observations = self._observe()
        elif self.autoreset_mode == "same_step":
            if done.any():
                final_obs = self._observe()
                infos = self._with_masks(infos)
                final = np.full(self.num_envs, None, dtype=object)
                final_info = np.full(self.num_envs, None, dtype=object)
                for index in np.flatnonzero(done):
                    final[index] = final_obs[index].copy()
                    final_info[index] = _info_at(infos, index)
                self._reset_envs(done, None)
                infos["final_obs"], infos["_final_obs"] = final, done.copy()
                infos["final_info"], infos["_final_info"] = final_info, done.copy()
            observations = self._observe()
        else:
            observations = self._observe()
        return observations, rewards, terminated, truncated, self._with_masks(infos)

    def render(self) -> Any:
        """Rendering is not implemented by the base class."""
        if self.render_mode is None:
            return None
        raise NotImplementedError(
            f"{type(self).__name__} does not render; render a single environment instead"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_actions(self, actions: Any) -> np.ndarray:
        space = self.action_space
        if isinstance(space, spaces.Box):
            array = np.asarray(actions, dtype=space.dtype)
        else:
            array = np.asarray(actions)
        expected = space.shape
        if expected is not None and array.shape != expected:
            if array.size == int(np.prod(expected)):
                array = array.reshape(expected)
            else:
                raise ValueError(f"actions must have shape {expected}, got {array.shape}")
        if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ValueError("actions contain NaN or infinite values")
        return array

    def _with_masks(self, infos: dict[str, Any]) -> dict[str, Any]:
        """Add the ``"_key"`` masks Gymnasium expects for every info key."""
        out: dict[str, Any] = {}
        for key, value in infos.items():
            if key.startswith("_"):
                out[key] = value
                continue
            out[key] = value
            if f"_{key}" not in infos:
                out[f"_{key}"] = np.ones(self.num_envs, dtype=bool)
        return out


def _info_at(infos: dict[str, Any], index: int) -> dict[str, Any]:
    """Info dictionary of one copy, extracted from batched infos."""
    single: dict[str, Any] = {}
    for key, value in infos.items():
        if key.startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if mask is not None and not mask[index]:
            continue
        single[key] = value[index] if isinstance(value, np.ndarray) else value
    return single
