"""Tk GUI for tuning the defaults of an ``argparse`` parser.

:class:`ParameterTuner` opens a window with one field per parser option
(check boxes for booleans, drop-down lists for ``choices``, text fields
otherwise). Values are validated with :func:`toolkit.parakit.parse_value`;
accepted values are written to timestamped JSON files and installed as the
parser's defaults when the window closes.

Tk is imported only when the window is opened, so this module (and the headless
API in :mod:`toolkit.parakit.params`) works on systems without Tk.

Window behaviour
----------------
* **Save Parameters** validates and saves the current values.
* **OK** saves and closes the window.
* **Auto-save** (toggle) saves every ``save_delay`` milliseconds.
* After ``inactivity_timeout`` seconds without keyboard or mouse activity the
  values are saved and the window closes (disable with ``inactivity_timeout=None``).
* **Browse** / **Reset to Default** change the output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from .params import (
    ValidationCallback,
    describe_type,
    format_value,
    get_parameters,
    parse_value,
    save_parameters,
    tunable_actions,
)

__all__ = ["TK_MISSING_MESSAGE", "ParameterAdjuster", "ParameterTuner"]

logger = logging.getLogger(__name__)

TK_MISSING_MESSAGE = (
    "The parakit GUI requires Tk (the 'tkinter' module), which is not available in this "
    "Python installation. Install Tk, for example 'sudo apt-get install python3-tk' "
    "(Debian/Ubuntu), 'sudo dnf install python3-tkinter' (Fedora), "
    "'brew install python-tk' (macOS with Homebrew), or 'conda install tk'. Without Tk, "
    "use the headless API: get_parameters, save_parameters, load_parameters and "
    "apply_parameters."
)


def _import_tk() -> tuple[Any, Any, Any, Any]:
    """Import tkinter lazily and return ``(tk, ttk, messagebox, filedialog)``."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise ImportError(TK_MISSING_MESSAGE) from exc
    return tk, ttk, messagebox, filedialog


class ParameterTuner:
    """GUI tuner for the defaults of an :class:`argparse.ArgumentParser`.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser whose defaults are tuned.
    save_delay : int, default 30000
        Auto-save interval in milliseconds.
    save_path : str or path-like, optional
        Directory for the timestamped JSON files (default: current working
        directory). It is created when the first file is written.
    auto_save_enabled : bool, default True
        Start with auto-save enabled.
    validation_callbacks : dict, optional
        Per-destination validators applied to converted values (see
        :data:`toolkit.parakit.params.ValidationCallback`).
    inactivity_timeout : float or None, default 10
        Seconds without user activity before the window saves and closes;
        ``None`` disables the automatic close.

    Attributes
    ----------
    args_defaults : dict
        Destination name to ``argparse.Action`` for every tunable option.
    last_saved_path : pathlib.Path or None
        File written by the most recent successful save.

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> _ = parser.add_argument("--lr", type=float, default=0.01)
    >>> parser = ParameterTuner.tune_parameters(parser)  # doctest: +SKIP
    >>> args = parser.parse_args()  # doctest: +SKIP
    """

    font_family = "Times New Roman"

    def __init__(
        self,
        parser: argparse.ArgumentParser,
        save_delay: int = 30000,
        save_path: str | Path | None = None,
        auto_save_enabled: bool = True,
        validation_callbacks: Mapping[str, ValidationCallback] | None = None,
        inactivity_timeout: float | None = 10,
    ) -> None:
        if not isinstance(parser, argparse.ArgumentParser):
            raise TypeError(
                f"parser must be an argparse.ArgumentParser, got {type(parser).__name__}"
            )
        if (
            isinstance(save_delay, bool)
            or not isinstance(save_delay, (int, float))
            or save_delay < 1
        ):
            raise ValueError(
                f"save_delay must be a positive number of milliseconds, got {save_delay!r}"
            )
        if inactivity_timeout is not None and not inactivity_timeout > 0:
            raise ValueError(
                f"inactivity_timeout must be a positive number of seconds or None, got {inactivity_timeout!r}"
            )
        self.parser = parser
        self.save_delay = int(save_delay)
        self.save_path = Path(save_path) if save_path is not None else self.get_default_save_path()
        self.auto_save_enabled = bool(auto_save_enabled)
        self.validation_callbacks: dict[str, ValidationCallback] = dict(validation_callbacks or {})
        self.inactivity_timeout = inactivity_timeout
        self.args_defaults = self._extract_parameter_info()
        unknown = sorted(set(self.validation_callbacks) - set(self.args_defaults))
        if unknown:
            warnings.warn(
                f"validation_callbacks for unknown parameters are ignored: {', '.join(unknown)}",
                stacklevel=2,
            )
        self.last_saved_path: Path | None = None
        self._accepted: dict[str, Any] | None = None

        # GUI state (created by tune()).
        self.root: Any = None
        self.entries: dict[str, Any] = {}
        self.status_label: Any = None
        self.save_path_var: Any = None
        self.auto_save_var: Any = None
        self._tk: Any = None
        self._messagebox: Any = None
        self._filedialog: Any = None
        self._auto_save_timer: Any = None
        self._inactivity_timer: Any = None
        self._is_saving = False
        self._ok_clicked = False
        self._last_activity = time.time()

    # ------------------------------------------------------------------ logic

    def _extract_parameter_info(self) -> dict[str, argparse.Action]:
        """Destination name to action for every tunable option of the parser."""
        return tunable_actions(self.parser)

    def _validate_parameter(self, key: str, value: Any) -> tuple[bool, Any, str]:
        """Convert and validate one value.

        Returns
        -------
        tuple
            ``(is_valid, converted_value, error_message)``; ``converted_value`` is
            ``None`` and ``error_message`` non-empty when invalid.
        """
        action = self.args_defaults.get(key)
        if action is None:
            return False, None, f"unknown parameter {key!r}"
        try:
            converted = parse_value(action, value, self.validation_callbacks.get(key))
        except ValueError as exc:
            return False, None, str(exc)
        return True, converted, ""

    def _collect_values(self) -> tuple[dict[str, Any], list[str]]:
        """Validated values of all fields (or the parser defaults before the GUI exists)."""
        if not self.entries:
            return get_parameters(self.parser), []
        values: dict[str, Any] = {}
        errors: list[str] = []
        for key, variable in self.entries.items():
            ok, converted, message = self._validate_parameter(key, variable.get())
            if ok:
                values[key] = converted
            else:
                errors.append(f"{key}: {message}")
        return values, errors

    @classmethod
    def get_default_save_path(cls) -> Path:
        """Default output directory: the current working directory."""
        return Path.cwd()

    def _save_parameters(self, auto: bool = False) -> bool:
        """Validate the current values and write them to a new JSON file.

        Parameters
        ----------
        auto : bool, default False
            True for timer-triggered saves: problems are reported in the status
            line instead of dialogs.

        Returns
        -------
        bool
            True if a file was written.
        """
        if self._is_saving:
            return False
        self._is_saving = True
        try:
            values, errors = self._collect_values()
            if errors:
                message = "Validation errors:\n" + "\n".join(errors)
                logger.warning("Parameters not saved. %s", message)
                self._set_status("Not saved: invalid values")
                if not auto:
                    self._show_message("Validation Error", message, "error")
                return False
            try:
                path = save_parameters(values, self.save_path)
            except OSError as exc:
                message = f"Failed to save parameters: {exc}"
                logger.error(message)
                self._set_status("Save failed")
                if not auto:
                    self._show_message("Save Error", message, "error")
                return False
            self.last_saved_path = path
            self._accepted = values
            logger.info("Parameters saved to %s", path)
            self._set_status(f"Saved: {path.name}")
            if not auto and not self._ok_clicked:
                self._show_message(
                    "Save Successful",
                    f"Parameters saved.\n\nFile: {path.name}\nLocation: {path.parent}",
                )
            return True
        finally:
            self._is_saving = False

    # -------------------------------------------------------------- GUI glue

    def _font(self, size: int, bold: bool = False) -> tuple[Any, ...]:
        return (self.font_family, size, "bold") if bold else (self.font_family, size)

    def _set_status(self, text: str) -> None:
        if self.status_label is not None:
            self.status_label.config(text=text)

    def _show_message(
        self, title: str, message: str, message_type: str = "info", auto_close: bool = True
    ) -> None:
        """Show a dialog; info messages close themselves after three seconds."""
        if self.root is None or self._messagebox is None:
            logger.info("%s: %s", title, message)
            return
        if message_type == "error":
            self._messagebox.showerror(title, message, parent=self.root)
        elif message_type == "warning":
            self._messagebox.showwarning(title, message, parent=self.root)
        elif auto_close:
            window = self._tk.Toplevel(self.root)
            window.title(title)
            window.resizable(False, False)
            window.transient(self.root)
            label = self._tk.Label(
                window, text=message, padx=20, pady=20, wraplength=360, font=self._font(10)
            )
            label.pack(expand=True, fill="both")
            window.after(3000, window.destroy)
        else:
            self._messagebox.showinfo(title, message, parent=self.root)

    def _save_and_close(self) -> None:
        """OK button: save, then close the window if the save succeeded."""
        self._ok_clicked = True
        if self._save_parameters():
            self._close()
        else:
            self._ok_clicked = False

    def _auto_save(self) -> None:
        """Timer callback: save and schedule the next auto-save."""
        self._auto_save_timer = None
        if self.root is None or not self.auto_save_enabled:
            return
        if self._save_parameters(auto=True):
            logger.info("Auto-save completed")
        self._auto_save_timer = self.root.after(self.save_delay, self._auto_save)

    def _tcl_error(self) -> tuple[type, ...]:
        """Exception type raised by Tk for operations on destroyed widgets."""
        return (self._tk.TclError,) if self._tk is not None else ()

    def _cancel_timers(self) -> None:
        if self.root is None:
            return
        for name in ("_auto_save_timer", "_inactivity_timer"):
            timer = getattr(self, name)
            if timer is not None:
                with contextlib.suppress(*self._tcl_error()):
                    self.root.after_cancel(timer)
                setattr(self, name, None)

    def _close(self) -> None:
        """Cancel timers and destroy the window (ends ``mainloop``)."""
        if self.root is None:
            return
        self._cancel_timers()
        with contextlib.suppress(*self._tcl_error()):  # already destroyed by the window manager
            self.root.destroy()
        self.root = None
        self.status_label = None

    def _browse_save_path(self) -> None:
        """Ask for a new output directory."""
        if self._filedialog is None:
            return
        new_path = self._filedialog.askdirectory(
            initialdir=str(self.save_path), title="Select Save Directory", parent=self.root
        )
        if new_path:
            self._set_save_path(Path(new_path), f"Save path updated: {new_path}")

    def _reset_save_path(self) -> None:
        """Reset the output directory to :meth:`get_default_save_path`."""
        self._set_save_path(self.get_default_save_path(), "Save path reset to default")

    def _set_save_path(self, path: Path, status: str) -> None:
        self.save_path = path
        if self.save_path_var is not None:
            self.save_path_var.set(str(path))
        self._set_status(status)

    def _on_activity(self, _event: Any = None) -> None:
        """Record user activity and restart the inactivity countdown."""
        self._last_activity = time.time()
        self._reset_inactivity_timer()

    def _reset_inactivity_timer(self) -> None:
        if self.root is None or self.inactivity_timeout is None:
            return
        if self._inactivity_timer is not None:
            self.root.after_cancel(self._inactivity_timer)
        delay_ms = int(self.inactivity_timeout * 1000)
        self._inactivity_timer = self.root.after(delay_ms, self._on_inactivity_timeout)

    def _on_inactivity_timeout(self) -> None:
        """Save (if the values are valid) and close after the inactivity timeout."""
        self._inactivity_timer = None
        if self.root is None:
            return
        logger.info("No activity for %s seconds; saving and closing", self.inactivity_timeout)
        self._set_status("Saving due to inactivity...")
        if not self._save_parameters(auto=True):
            logger.warning("Closing without saving: the current values are invalid")
        self.root.after(500, self._close)

    def _toggle_auto_save(self) -> None:
        """Checkbox callback: start or stop the auto-save timer."""
        self.auto_save_enabled = bool(self.auto_save_var.get())
        if self.root is None:
            return
        if self.auto_save_enabled and self._auto_save_timer is None:
            self._auto_save_timer = self.root.after(self.save_delay, self._auto_save)
            self._set_status("Auto-save enabled")
        elif not self.auto_save_enabled and self._auto_save_timer is not None:
            self.root.after_cancel(self._auto_save_timer)
            self._auto_save_timer = None
            self._set_status("Auto-save disabled")

    def _create_gui(self) -> None:
        """Build the window (requires Tk and a display)."""
        tk, ttk, messagebox, filedialog = _import_tk()
        self._tk, self._messagebox, self._filedialog = tk, messagebox, filedialog
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError(
                f"Cannot open the parameter tuner window ({exc}). A graphical display is "
                "required; on headless systems use the headless parakit API instead."
            ) from exc
        self.root = root
        root.title("Parameter Tuner")
        root.geometry("720x600")
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        main = ttk.Frame(root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Output directory.
        save_frame = ttk.LabelFrame(main, text="Save Configuration", padding=10)
        save_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        save_frame.grid_columnconfigure(1, weight=1)
        tk.Label(save_frame, text="Save Path:", font=self._font(10, True)).grid(
            row=0, column=0, sticky="w", padx=(0, 5)
        )
        self.save_path_var = tk.StringVar(master=root, value=str(self.save_path))
        ttk.Entry(save_frame, textvariable=self.save_path_var, state="readonly", width=50).grid(
            row=0, column=1, sticky="ew", padx=(0, 5)
        )
        tk.Button(
            save_frame, text="Browse", font=self._font(9, True), command=self._browse_save_path
        ).grid(row=0, column=2, padx=(0, 5))
        tk.Button(
            save_frame,
            text="Reset to Default",
            font=self._font(9, True),
            command=self._reset_save_path,
        ).grid(row=0, column=3)

        # Scrollable parameter list.
        canvas = tk.Canvas(main, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main, orient="vertical", command=canvas.yview)
        fields = ttk.Frame(canvas)
        fields.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=fields, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        fields.grid_columnconfigure(1, weight=1)

        self.entries = {}
        current = get_parameters(self.parser)
        for row, (key, action) in enumerate(self.args_defaults.items()):
            tk.Label(fields, text=f"{key}:", font=self._font(11, True)).grid(
                row=2 * row, column=0, sticky="w", padx=(10, 5), pady=(10, 2)
            )
            if action.help and action.help != argparse.SUPPRESS:
                tk.Label(
                    fields,
                    text=action.help,
                    font=self._font(9),
                    fg="gray",
                    wraplength=480,
                    justify="left",
                ).grid(row=2 * row, column=1, columnspan=2, sticky="w", padx=(5, 10), pady=(10, 2))
            self.entries[key] = self._create_field(
                tk, ttk, fields, 2 * row + 1, action, current[key]
            )

        # Buttons and status line.
        buttons = ttk.Frame(root, padding=(10, 0, 10, 10))
        buttons.grid(row=1, column=0, sticky="ew")
        tk.Button(
            buttons,
            text="Save Parameters",
            font=self._font(10, True),
            command=self._save_parameters,
        ).pack(side="left", padx=(0, 10))
        tk.Button(buttons, text="OK", font=self._font(10, True), command=self._save_and_close).pack(
            side="left", padx=(0, 10)
        )
        self.auto_save_var = tk.BooleanVar(master=root, value=self.auto_save_enabled)
        tk.Checkbutton(
            buttons,
            text="Auto-save",
            font=self._font(9),
            variable=self.auto_save_var,
            command=self._toggle_auto_save,
        ).pack(side="left", padx=(0, 10))
        self.status_label = tk.Label(buttons, text="Ready", font=self._font(8))
        self.status_label.pack(side="right")

        # Any keyboard or mouse activity restarts the inactivity countdown.
        for sequence in ("<Key>", "<Button>", "<Motion>", "<MouseWheel>"):
            root.bind_all(sequence, self._on_activity, add="+")

        if self.auto_save_enabled:
            self._auto_save_timer = root.after(self.save_delay, self._auto_save)
        self._reset_inactivity_timer()

    def _create_field(
        self, tk: Any, ttk: Any, parent: Any, row: int, action: argparse.Action, value: Any
    ) -> Any:
        """Create the input widget for one option and return its variable."""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 8))
        type_name = describe_type(action)
        if type_name == "bool":
            variable = tk.BooleanVar(master=self.root, value=bool(value))
            ttk.Checkbutton(frame, variable=variable).pack(side="left")
        elif action.choices and action.nargs in (None, "?"):
            variable = tk.StringVar(master=self.root, value=format_value(action, value))
            options = [format_value(action, choice) for choice in action.choices]
            if action.nargs == "?" or action.default is None:
                options.append("None")
            ttk.Combobox(
                frame, textvariable=variable, values=options, state="readonly", width=57
            ).pack(side="left", fill="x", expand=True)
        else:
            variable = tk.StringVar(master=self.root, value=format_value(action, value))
            ttk.Entry(frame, textvariable=variable, width=60).pack(
                side="left", fill="x", expand=True
            )
        tk.Label(frame, text=f"({type_name})", font=self._font(8), fg="blue").pack(
            side="right", padx=(5, 0)
        )
        return variable

    # ------------------------------------------------------------ public API

    def tune(self) -> argparse.ArgumentParser:
        """Open the window and block until it closes.

        The values of the last successful save (manual, OK, auto-save or
        inactivity save) are installed with ``parser.set_defaults``; if nothing
        was saved the parser is unchanged.

        Returns
        -------
        argparse.ArgumentParser
            The same parser, with updated defaults.

        Raises
        ------
        ImportError
            If Tk is not installed (the message explains how to install it).
        RuntimeError
            If no display is available.
        """
        self._accepted = None
        self._ok_clicked = False
        self._create_gui()
        try:
            self.root.mainloop()
        finally:
            self._close()
            self.entries = {}
        if self._accepted is not None:
            self.parser.set_defaults(**self._accepted)
        return self.parser

    @classmethod
    def tune_parameters(
        cls,
        parser: argparse.ArgumentParser,
        save_delay: int = 30000,
        save_path: str | Path | None = None,
        auto_save_enabled: bool = True,
        validation_callbacks: Mapping[str, Callable[[Any], Any]] | None = None,
        inactivity_timeout: float | None = 10,
    ) -> argparse.ArgumentParser:
        """Create a :class:`ParameterTuner`, open it and return the updated parser."""
        tuner = cls(
            parser=parser,
            save_delay=save_delay,
            save_path=save_path,
            auto_save_enabled=auto_save_enabled,
            validation_callbacks=validation_callbacks,
            inactivity_timeout=inactivity_timeout,
        )
        return tuner.tune()


ParameterAdjuster = ParameterTuner
"""Backward-compatible alias of :class:`ParameterTuner`."""
