"""Tune the defaults of an argparse parser with toolkit.parakit.

The demo builds a typical training parser (floats, ints, choices, a list with
``nargs='+'``, booleans and an optional path) and then:

1. applies parameter values from a JSON file (``--load``) and/or ``--set KEY=VALUE``
   overrides, validated against the parser (types, choices, nargs) plus two
   custom validation callbacks;
2. optionally opens the Tk tuner window (``--gui``; requires Tk and a display),
   which saves timestamped JSON files and installs the accepted values as the
   parser's new defaults;
3. saves the resulting parameters to ``--save-dir`` (default ``renders/parakit``)
   and prints the configuration the training script would run with.

Examples
--------
Save the defaults, then reload the newest file with one override::

    python examples/toolkit/parakit_demo.py
    python examples/toolkit/parakit_demo.py --load renders/parakit --set batch_size=128

Edit the values in a window (auto-closes after 30 s without activity)::

    python examples/toolkit/parakit_demo.py --gui --inactivity-timeout 30
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from toolkit.parakit import (
    ParameterTuner,
    apply_parameters,
    get_parameters,
    load_parameters,
    save_parameters,
)


def build_training_parser() -> argparse.ArgumentParser:
    """The parser whose defaults are tuned."""
    parser = argparse.ArgumentParser(description="Example training configuration")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate in (0, 1]")
    parser.add_argument("--batch_size", type=int, default=32, help="Positive even batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam", help="Optimizer"
    )
    parser.add_argument(
        "--hidden_sizes", type=int, nargs="+", default=[256, 256], help="Hidden layer widths"
    )
    parser.add_argument("--layer_norm", action="store_true", help="Use layer normalisation")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to resume from")
    return parser


def validate_learning_rate(value: float) -> tuple[bool, str]:
    """Custom validation callback: 0 < lr <= 1."""
    return 0.0 < value <= 1.0, "learning rate must be in (0, 1]"


def validate_batch_size(value: int) -> tuple[bool, str]:
    """Custom validation callback: positive and even."""
    return value > 0 and value % 2 == 0, "batch size must be a positive even number"


VALIDATORS = {"learning_rate": validate_learning_rate, "batch_size": validate_batch_size}


def parse_overrides(items: Sequence[str]) -> dict[str, str]:
    """Turn ``KEY=VALUE`` strings into a dict (values stay text; parakit converts them)."""
    overrides: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = value
    return overrides


def format_table(values: dict[str, Any]) -> list[str]:
    width = max(len(key) for key in values)
    return [f"  {key:<{width}}  {value!r}" for key, value in values.items()]


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--gui", action="store_true", help="open the Tk tuner window")
    cli.add_argument("--load", type=Path, help="JSON file, or directory with parameters_*.json")
    cli.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override one value"
    )
    cli.add_argument(
        "--save-dir", type=Path, default=Path("renders/parakit"), help="output directory"
    )
    cli.add_argument(
        "--inactivity-timeout",
        type=float,
        default=60.0,
        help="GUI auto-close after this many idle seconds (0 disables)",
    )
    args = cli.parse_args(argv)

    parser = build_training_parser()
    print("Initial defaults:")
    print("\n".join(format_table(get_parameters(parser))))

    try:
        if args.load is not None:
            apply_parameters(parser, load_parameters(args.load), validation_callbacks=VALIDATORS)
            print(f"\nApplied parameters from {args.load}")
        if args.set:
            apply_parameters(parser, parse_overrides(args.set), validation_callbacks=VALIDATORS)
            print(f"Applied {len(args.set)} override(s)")
    except (OSError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2

    if args.gui:
        timeout = args.inactivity_timeout if args.inactivity_timeout > 0 else None
        tuner = ParameterTuner(
            parser,
            save_path=args.save_dir,
            validation_callbacks=VALIDATORS,
            inactivity_timeout=timeout,
        )
        try:
            tuner.tune()
        except (ImportError, RuntimeError) as exc:
            print(f"\nCannot open the GUI: {exc}", file=sys.stderr)
            return 1
        if tuner.last_saved_path is not None:
            print(f"\nGUI saved {tuner.last_saved_path}")
        else:
            print("\nGUI closed without saving; defaults unchanged")
    else:
        path = save_parameters(get_parameters(parser), args.save_dir)
        print(f"\nSaved parameters to {path}")

    config = parser.parse_args([])
    print("\nConfiguration used for training:")
    print("\n".join(format_table(vars(config))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
