"""Tests for toolkit.parakit (headless parameter API and the Tk tuner logic).

The GUI is exercised with a stand-in ``tkinter`` module, so these tests run on
headless machines and Python builds without Tk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from toolkit.parakit import (
    TK_MISSING_MESSAGE,
    ParameterAdjuster,
    ParameterTuner,
    apply_parameters,
    describe_type,
    format_value,
    get_parameters,
    load_parameters,
    parse_bool,
    parse_value,
    save_parameters,
    tunable_actions,
)

TIMESTAMPED = re.compile(r"parameters_\d{8}_\d{6}(_\d+)?\.json")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="test parser")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_cuda", action="store_false", dest="cuda")
    parser.add_argument("--checkpoint", default=None, help="Optional path")
    parser.add_argument("--name", default="run")
    return parser


def action_of(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    return tunable_actions(parser)[dest]


# ---------------------------------------------------------------------------
# Import behaviour
# ---------------------------------------------------------------------------


def test_import_without_tkinter_and_without_logging_side_effects():
    code = textwrap.dedent(
        """
        import logging, sys
        sys.modules["tkinter"] = None  # simulate a Python build without Tk
        import toolkit.parakit as pk
        assert not logging.getLogger().handlers, "root logger was configured"
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--x", type=int, default=1)
        try:
            pk.ParameterTuner(parser).tune()
        except ImportError as exc:
            assert "python3-tk" in str(exc)
            print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_tune_raises_clear_import_error_without_tk(monkeypatch):
    monkeypatch.setitem(sys.modules, "tkinter", None)
    tuner = ParameterTuner(make_parser())
    with pytest.raises(ImportError, match="headless API") as info:
        tuner.tune()
    assert str(info.value) == TK_MISSING_MESSAGE


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("true", True),
        ("False", False),
        (" YES ", True),
        ("no", False),
        ("on", True),
        ("OFF", False),
        ("1", True),
        ("0", False),
        (1, True),
        (0, False),
        (True, True),
    ],
)
def test_parse_bool(text, expected):
    assert parse_bool(text) is expected


@pytest.mark.parametrize("text", ["maybe", "", 2, None, "2"])
def test_parse_bool_rejects(text):
    with pytest.raises(ValueError, match="invalid boolean"):
        parse_bool(text)


def test_boolean_actions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", action="store_true")
    parser.add_argument("--b", action="store_false")
    parser.add_argument("--c", type=bool, default=True)
    parser.add_argument("--d", default=False)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--e", action=argparse.BooleanOptionalAction, default=True)
    for dest, action in tunable_actions(parser).items():
        assert parse_value(action, "False") is False, dest
        assert parse_value(action, "yes") is True, dest
        assert describe_type(action) == "bool", dest
        with pytest.raises(ValueError):
            parse_value(action, "sometimes")


def test_scalar_types_and_inference():
    parser = make_parser()
    parser.add_argument("--untyped_float", default=0.5)
    parser.add_argument("--count", "-c", action="count", default=0)
    assert parse_value(action_of(parser, "learning_rate"), " 3e-4 ") == pytest.approx(3e-4)
    assert parse_value(action_of(parser, "batch_size"), "42") == 42
    assert parse_value(action_of(parser, "untyped_float"), "0.25") == 0.25
    assert parse_value(action_of(parser, "count"), "3") == 3
    assert parse_value(action_of(parser, "name"), " spaced name ") == " spaced name "
    # Already-typed values (for example from JSON).
    assert parse_value(action_of(parser, "batch_size"), 64.0) == 64
    assert parse_value(action_of(parser, "learning_rate"), 1) == 1.0
    with pytest.raises(ValueError, match="not an integer"):
        parse_value(action_of(parser, "batch_size"), 64.5)
    with pytest.raises(ValueError, match="invalid literal"):
        parse_value(action_of(parser, "batch_size"), "not_a_number")
    with pytest.raises(ValueError, match="invalid int value"):
        parse_value(action_of(parser, "batch_size"), True)


def test_custom_type_callable():
    def positive_int(text):
        value = int(text)
        if value <= 0:
            raise argparse.ArgumentTypeError("must be > 0")
        return value

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=positive_int, default=1)
    action = action_of(parser, "k")
    assert parse_value(action, "5") == 5
    with pytest.raises(ValueError, match="must be > 0"):
        parse_value(action, "-2")
    with pytest.raises(ValueError, match="must be > 0"):
        parse_value(action, -2)


def str2bool(text):
    """Typical command-line boolean converter (fails on non-strings)."""
    if text.lower() in ("yes", "true", "t", "1"):
        return True
    if text.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def test_custom_type_callables_accept_json_values():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--hexv", type=lambda s: int(s, 0), default=16)
    parser.add_argument("--path", type=Path, default=Path("runs/a b"))
    a = tunable_actions(parser)
    assert parse_value(a["use_gpu"], True) is True
    assert parse_value(a["use_gpu"], False) is False
    assert parse_value(a["use_gpu"], "yes") is True
    assert parse_value(a["hexv"], 32) == 32
    assert parse_value(a["hexv"], "0x20") == 32
    with pytest.raises(ValueError, match="Boolean value expected"):
        parse_value(a["use_gpu"], 5)
    with pytest.raises(ValueError, match="invalid value 1.5"):
        parse_value(a["hexv"], 1.5)

    def attribute_error(text):
        return text.missing_method()

    parser.add_argument("--broken", type=attribute_error, default=None)
    with pytest.raises(ValueError, match="invalid attribute_error value 'x'"):
        parse_value(tunable_actions(parser)["broken"], "x")
    with pytest.raises(ValueError, match="broken:"):
        apply_parameters(parser, {"broken": "x"})


def test_custom_type_callables_survive_save_load_apply(tmp_path):
    def make():
        parser = argparse.ArgumentParser()
        parser.add_argument("--use_gpu", type=str2bool, default=False)
        parser.add_argument("--hexv", type=lambda s: int(s, 0), default=16)
        return parser

    source = apply_parameters(make(), {"use_gpu": "true", "hexv": "0x40"})
    path = save_parameters(get_parameters(source), tmp_path)
    assert load_parameters(path) == {"use_gpu": True, "hexv": 64}
    fresh = apply_parameters(make(), load_parameters(path))
    assert vars(fresh.parse_args([])) == {"use_gpu": True, "hexv": 64}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("[3, 4.5]", [3.0, 4.5]),
        ("3, 4.5", [3.0, 4.5]),
        ("3 4.5  6", [3.0, 4.5, 6.0]),
        ("(1, 2)", [1.0, 2.0]),
        ([1, 2], [1.0, 2.0]),
        ("7", [7.0]),
    ],
)
def test_nargs_lists(text, expected):
    action = action_of(make_parser(), "weights")
    assert parse_value(action, text) == expected


def test_nargs_counts_and_list_actions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plus", type=int, nargs="+", default=[1])
    parser.add_argument("--star", nargs="*", default=[])
    parser.add_argument("--pair", type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument("--opt", type=int, nargs="?", default=3)
    parser.add_argument("--tag", action="append", default=None)
    parser.add_argument("--words", nargs="+", default=["a b", "c"])
    a = tunable_actions(parser)
    with pytest.raises(ValueError, match="at least one"):
        parse_value(a["plus"], "")
    assert parse_value(a["star"], "") == []
    assert parse_value(a["star"], "x, y") == ["x", "y"]
    with pytest.raises(ValueError, match="exactly 2"):
        parse_value(a["pair"], "1 2 3")
    assert parse_value(a["pair"], "[0.5, 1.5]") == [0.5, 1.5]
    assert parse_value(a["opt"], "None") is None
    assert parse_value(a["opt"], "4") == 4
    assert parse_value(a["tag"], "u,v") == ["u", "v"]
    assert parse_value(a["tag"], "") is None
    assert parse_value(a["words"], format_value(a["words"], ["a b", "c"])) == ["a b", "c"]
    with pytest.raises(ValueError, match="expected a single value"):
        parse_value(a["opt"], [1, 2])
    with pytest.raises(ValueError, match="cannot parse list"):
        parse_value(a["plus"], "[1, 2")


def test_append_and_extend_with_nargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", action="append", type=int, nargs=2, default=[[1, 2]])
    parser.add_argument("--group", action="append", nargs="+", default=None)
    parser.add_argument("--ext", action="extend", type=int, nargs=2, default=[1, 2, 3])
    parser.add_argument("--ext_plus", action="extend", nargs="+", default=[])
    parser.add_argument("--pick", action="append", nargs=2, choices=["a", "b"], default=[])
    a = tunable_actions(parser)
    assert describe_type(a["point"]) == "list[list[int]]"
    assert describe_type(a["ext"]) == "list[int]"
    assert parse_value(a["point"], [[3, 4], [5, 6]]) == [[3, 4], [5, 6]]
    assert parse_value(a["point"], "[[3, 4], [5, 6]]") == [[3, 4], [5, 6]]
    assert parse_value(a["point"], "3 4 5 6") == [[3, 4], [5, 6]]  # like --point 3 4 --point 5 6
    with pytest.raises(ValueError, match="exactly 2 values, got 3"):
        parse_value(a["point"], "[[1, 2, 3]]")
    with pytest.raises(ValueError, match="groups of 2"):
        parse_value(a["point"], "1 2 3")
    assert parse_value(a["group"], '[["x"], ["y", "z"]]') == [["x"], ["y", "z"]]
    with pytest.raises(ValueError, match="list of value lists"):
        parse_value(a["group"], "x y")
    assert parse_value(a["ext"], "1 2 3 4 5") == [1, 2, 3, 4, 5]  # accumulated values
    assert parse_value(a["ext_plus"], "[]") == []
    assert parse_value(a["pick"], [["a", "b"]]) == [["a", "b"]]
    with pytest.raises(ValueError, match="'c' is not a valid choice"):
        parse_value(a["pick"], [["a", "c"]])
    for value in ([[7, 8]], [], [[1, 2], [3, 4]]):
        assert parse_value(a["point"], format_value(a["point"], value)) == value
    assert parse_value(a["ext"], format_value(a["ext"], [1, 2, 3])) == [1, 2, 3]
    args = apply_parameters(parser, {"point": [[9, 9]], "ext": [4]}).parse_args(["--point", "0", "1"])
    assert args.point == [[9, 9], [0, 1]] and args.ext == [4]


def test_format_value_writes_non_json_list_elements_as_strings():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, nargs="+", default=[Path("/tmp/a b"), Path("c")])
    parser.add_argument("--nested", action="append", type=Path, nargs="+", default=None)
    a = tunable_actions(parser)
    text = format_value(a["paths"], [Path("/tmp/a b"), Path("c")])
    assert json.loads(text) == ["/tmp/a b", "c"]
    assert parse_value(a["paths"], text) == [Path("/tmp/a b"), Path("c")]
    nested = [[Path("x y")], [Path("z"), Path("w")]]
    assert parse_value(a["nested"], format_value(a["nested"], nested)) == nested


def test_none_handling():
    parser = make_parser()
    checkpoint = action_of(parser, "checkpoint")
    for text in ("None", "null", "", None):
        assert parse_value(checkpoint, text) is None
    assert parse_value(checkpoint, "model.pt") == "model.pt"
    with pytest.raises(ValueError, match="None is not allowed"):
        parse_value(action_of(parser, "batch_size"), "None")
    with pytest.raises(ValueError, match="None is not allowed"):
        parse_value(action_of(parser, "learning_rate"), None)
    # A string option with a non-None default keeps the literal text.
    assert parse_value(action_of(parser, "name"), "None") == "None"
    assert parse_value(action_of(parser, "name"), "") == ""


def test_choices():
    parser = make_parser()
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--modes", nargs="+", choices=["x", "y"], default=["x"])
    parser.add_argument("--untyped_int_choice", choices=[1, 2], default=1)
    assert parse_value(action_of(parser, "optimizer"), "sgd") == "sgd"
    with pytest.raises(ValueError, match="must be one of"):
        parse_value(action_of(parser, "optimizer"), "rmsprop")
    assert parse_value(action_of(parser, "level"), "2") == 2
    with pytest.raises(ValueError, match="must be one of"):
        parse_value(action_of(parser, "level"), "4")
    assert parse_value(action_of(parser, "modes"), "x y") == ["x", "y"]
    with pytest.raises(ValueError, match="'z' is not a valid choice"):
        parse_value(action_of(parser, "modes"), "x z")
    assert parse_value(action_of(parser, "untyped_int_choice"), "2") == 2


def test_validation_callbacks_protocols():
    action = action_of(make_parser(), "learning_rate")
    assert parse_value(action, "0.5", lambda x: 0 < x < 1) == 0.5
    with pytest.raises(ValueError, match="validation failed"):
        parse_value(action, "2.0", lambda x: 0 < x < 1)
    with pytest.raises(ValueError, match="too large"):
        parse_value(action, "2.0", lambda x: (x < 1, "too large"))

    def legacy(value):  # (is_valid, converted_value, error_message)
        val = float(value)
        return (True, val, "") if val > 0 else (False, None, "Value must be positive")

    assert parse_value(action, "5.0", legacy) == 5.0
    with pytest.raises(ValueError, match="must be positive"):
        parse_value(action, "-1.0", legacy)

    def raising(value):
        raise ValueError("custom failure")

    with pytest.raises(ValueError, match="custom failure"):
        parse_value(action, "0.1", raising)
    checkpoint = action_of(make_parser(), "checkpoint")
    assert parse_value(checkpoint, "None", raising) is None  # validators skip None


def test_format_value_round_trip():
    parser = make_parser()
    parser.add_argument("--pair", type=int, nargs=2, default=[3, 4])
    for dest, action in tunable_actions(parser).items():
        default = action.default
        assert parse_value(action, format_value(action, default)) == default, dest
    assert format_value(action_of(parser, "weights"), [1.0, 2.5]) == "[1.0, 2.5]"
    assert format_value(action_of(parser, "checkpoint"), None) == "None"


def test_describe_type():
    parser = make_parser()
    assert describe_type(action_of(parser, "learning_rate")) == "float"
    assert describe_type(action_of(parser, "weights")) == "list[float]"
    assert describe_type(action_of(parser, "checkpoint")) == "str or None"
    assert describe_type(action_of(parser, "verbose")) == "bool"


# ---------------------------------------------------------------------------
# get / save / load / apply
# ---------------------------------------------------------------------------


def test_get_parameters():
    parser = make_parser()
    parser.add_argument("--version", action="version", version="1.0")
    parser.add_argument("--hidden", default=argparse.SUPPRESS)
    params = get_parameters(parser)
    assert list(params) == [
        "learning_rate",
        "batch_size",
        "optimizer",
        "weights",
        "verbose",
        "cuda",
        "checkpoint",
        "name",
    ]
    assert params["cuda"] is True and params["verbose"] is False
    params["weights"].append(99.0)
    assert get_parameters(parser)["weights"] == [1.0, 2.0]
    with pytest.raises(TypeError):
        get_parameters({"not": "a parser"})


def test_save_into_directory_uses_timestamped_names(tmp_path):
    target = tmp_path / "nested" / "deep"
    first = save_parameters({"a": 1}, target)
    second = save_parameters({"a": 2}, target)
    assert target.is_dir()
    assert first.parent == target and TIMESTAMPED.fullmatch(first.name)
    assert TIMESTAMPED.fullmatch(second.name) and first != second
    assert json.loads(first.read_text()) == {"a": 1}
    assert load_parameters(target) == {"a": 2}  # newest file in the directory


def test_load_newest_file_orders_counters_numerically(tmp_path):
    stamp = "parameters_20250101_120000"
    names = [f"{stamp}.json"] + [f"{stamp}_{i}.json" for i in range(1, 12)]
    for i, name in enumerate(names):
        (tmp_path / name).write_text(json.dumps({"i": i}))
    (tmp_path / "parameters_best.json").write_text(json.dumps({"i": "best"}))
    (tmp_path / "parameters_20241231_235959_99.json").write_text(json.dumps({"i": "older"}))
    assert load_parameters(tmp_path) == {"i": 11}  # "_11" is newer than "_9"
    for name in names:
        (tmp_path / name).unlink()
    assert load_parameters(tmp_path) == {"i": "older"}  # timestamped names rank first


def test_save_many_times_in_one_directory_loads_the_last(tmp_path):
    paths = [save_parameters({"i": i}, tmp_path) for i in range(12)]
    assert len(set(paths)) == 12
    assert load_parameters(tmp_path) == {"i": 11}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_saved_file_has_default_permissions(tmp_path):
    old = os.umask(0o027)
    try:
        path = save_parameters({"a": 1}, tmp_path / "run.json")
        timestamped = save_parameters({"a": 1}, tmp_path)
    finally:
        os.umask(old)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert stat.S_IMODE(timestamped.stat().st_mode) == 0o640
    assert not list(tmp_path.glob(".*.tmp"))


def test_save_to_explicit_file_and_non_json_values(tmp_path):
    path = save_parameters({"out": Path("/tmp/x"), "values": (1, 2)}, tmp_path / "sub" / "run.json")
    assert path == tmp_path / "sub" / "run.json"
    assert load_parameters(path) == {"out": "/tmp/x", "values": [1, 2]}
    assert not list(path.parent.glob("*.tmp"))


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_parameters(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        load_parameters(tmp_path)
    (tmp_path / "list.json").write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        load_parameters(tmp_path / "list.json")
    (tmp_path / "bad.json").write_text("{nope")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_parameters(tmp_path / "bad.json")


def test_apply_parameters_sets_defaults():
    parser = make_parser()
    result = apply_parameters(
        parser,
        {
            "learning_rate": "0.1",
            "weights": "[0.5, 0.25]",
            "verbose": "True",
            "cuda": False,
            "checkpoint": "m.pt",
        },
    )
    assert result is parser
    args = parser.parse_args([])
    assert args.learning_rate == 0.1 and args.weights == [0.5, 0.25]
    assert args.verbose is True and args.cuda is False and args.checkpoint == "m.pt"
    assert parser.parse_args(["--learning_rate", "0.2"]).learning_rate == 0.2  # CLI still wins


def test_apply_parameters_strict_and_errors():
    parser = make_parser()
    with pytest.raises(ValueError, match="Unknown parameter.*bogus.*valid parameters are"):
        apply_parameters(parser, {"bogus": 1, "batch_size": 8})
    assert parser.parse_args([]).batch_size == 32  # unchanged
    apply_parameters(parser, {"bogus": 1, "batch_size": 8}, strict=False)
    assert parser.parse_args([]).batch_size == 8
    with pytest.raises(ValueError) as info:
        apply_parameters(
            parser, {"batch_size": "x", "optimizer": "rmsprop", "learning_rate": "0.3"}
        )
    message = str(info.value)
    assert "batch_size:" in message and "optimizer:" in message
    assert parser.parse_args([]).learning_rate == 0.01  # nothing applied on error
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_parameters(
            parser,
            {"learning_rate": 5},
            validation_callbacks={"learning_rate": lambda v: (0 < v < 1, "between 0 and 1")},
        )


def test_save_load_apply_round_trip(tmp_path):
    source = make_parser()
    apply_parameters(
        source, {"batch_size": 128, "weights": [3.0], "optimizer": "sgd", "verbose": True}
    )
    path = save_parameters(get_parameters(source), tmp_path)
    fresh = make_parser()
    apply_parameters(fresh, load_parameters(path))
    assert vars(fresh.parse_args([])) == vars(source.parse_args([]))


# ---------------------------------------------------------------------------
# ParameterTuner (headless parts)
# ---------------------------------------------------------------------------


def test_tuner_construction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parser = make_parser()
    tuner = ParameterTuner(parser)
    assert tuner.parser is parser
    assert tuner.save_delay == 30000 and tuner.auto_save_enabled is True
    assert tuner.inactivity_timeout == 10
    assert "learning_rate" in tuner.args_defaults and "help" not in tuner.args_defaults
    assert tuner.save_path == ParameterTuner.get_default_save_path() == tmp_path
    nested = tmp_path / "a" / "b"
    tuner = ParameterTuner(parser, save_path=str(nested), inactivity_timeout=None)
    assert tuner.save_path == nested and not nested.exists()  # created lazily on save
    assert ParameterAdjuster is ParameterTuner


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(save_delay=0),
        dict(save_delay=True),
        dict(inactivity_timeout=0),
        dict(inactivity_timeout=-5),
    ],
)
def test_tuner_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        ParameterTuner(make_parser(), **kwargs)


def test_tuner_rejects_non_parser():
    with pytest.raises(TypeError):
        ParameterTuner({"lr": 0.1})


def test_tuner_warns_on_unknown_callback():
    with pytest.warns(UserWarning, match="unknown parameters"):
        ParameterTuner(make_parser(), validation_callbacks={"nope": bool})


def test_tuner_validate_parameter():
    def validate_positive(value):
        try:
            val = float(value)
        except (TypeError, ValueError):
            return False, None, "Value must be a number"
        return (True, val, "") if val > 0 else (False, None, "Value must be positive")

    tuner = ParameterTuner(make_parser(), validation_callbacks={"learning_rate": validate_positive})
    assert tuner._validate_parameter("batch_size", "42") == (True, 42, "")
    assert tuner._validate_parameter("learning_rate", "3.14") == (True, 3.14, "")
    assert tuner._validate_parameter("optimizer", "sgd") == (True, "sgd", "")
    ok, value, error = tuner._validate_parameter("batch_size", "not_a_number")
    assert not ok and value is None and "invalid literal" in error
    ok, value, error = tuner._validate_parameter("optimizer", "d")
    assert not ok and value is None and "must be one of" in error
    ok, _, error = tuner._validate_parameter("learning_rate", "-1.0")
    assert not ok and "must be positive" in error
    ok, _, error = tuner._validate_parameter("learning_rate", "not_a_number")
    assert not ok and "invalid float value" in error
    assert tuner._validate_parameter("unknown", "1")[0] is False


def test_tuner_headless_save_and_reset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tuner = ParameterTuner(make_parser(), save_path=tmp_path / "out")
    assert tuner._save_parameters() is True
    assert TIMESTAMPED.fullmatch(tuner.last_saved_path.name)
    assert load_parameters(tuner.last_saved_path)["batch_size"] == 32
    tuner._reset_save_path()
    assert tuner.save_path == tmp_path


# ---------------------------------------------------------------------------
# ParameterTuner GUI flow with a stand-in tkinter module
# ---------------------------------------------------------------------------


class _Var:
    """Minimal stand-in for tk.StringVar / tk.BooleanVar."""

    def __init__(self, master=None, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


@pytest.fixture
def fake_tk(monkeypatch):
    tk = MagicMock(name="tkinter")
    tk.StringVar = _Var
    tk.BooleanVar = _Var
    tk.TclError = type("TclError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    return tk


def _run_gui(fake_tk, tuner, user_actions):
    root = fake_tk.Tk.return_value
    root.mainloop.side_effect = lambda: user_actions(tuner, root)
    return tuner.tune(), root


def test_gui_ok_saves_applies_and_closes(fake_tk, tmp_path):
    parser = make_parser()
    tuner = ParameterTuner(parser, save_path=tmp_path, auto_save_enabled=False)

    def user(tuner, root):
        assert isinstance(tuner.entries["verbose"].get(), bool)
        assert tuner.entries["weights"].get() == "[1.0, 2.0]"
        tuner.entries["learning_rate"].set("0.5")
        tuner.entries["weights"].set("4, 5")
        tuner.entries["verbose"].set(True)
        tuner.entries["checkpoint"].set("None")
        tuner._save_and_close()

    result, root = _run_gui(fake_tk, tuner, user)
    assert result is parser
    root.destroy.assert_called()
    assert tuner.root is None and tuner.entries == {}
    args = parser.parse_args([])
    assert args.learning_rate == 0.5 and args.weights == [4.0, 5.0] and args.verbose is True
    saved = load_parameters(tuner.last_saved_path)
    assert saved["learning_rate"] == 0.5 and saved["checkpoint"] is None


def test_gui_close_without_saving_keeps_defaults(fake_tk, tmp_path):
    parser = make_parser()
    tuner = ParameterTuner(parser, save_path=tmp_path, auto_save_enabled=False)

    def user(tuner, root):
        tuner.entries["batch_size"].set("7")
        tuner._close()  # window closed by the window manager

    _run_gui(fake_tk, tuner, user)
    assert parser.parse_args([]).batch_size == 32
    assert tuner.last_saved_path is None and not list(tmp_path.iterdir())


def test_gui_invalid_values_block_ok(fake_tk, tmp_path):
    parser = make_parser()
    tuner = ParameterTuner(parser, save_path=tmp_path, auto_save_enabled=False)

    def user(tuner, root):
        tuner.entries["batch_size"].set("many")
        tuner._save_and_close()
        assert tuner.root is not None  # still open
        fake_tk.messagebox.showerror.assert_called_once()
        assert "batch_size" in fake_tk.messagebox.showerror.call_args[0][1]
        tuner.entries["batch_size"].set("16")
        tuner._save_and_close()

    _run_gui(fake_tk, tuner, user)
    assert parser.parse_args([]).batch_size == 16


def test_gui_inactivity_timeout_saves_and_closes(fake_tk, tmp_path):
    parser = make_parser()
    tuner = ParameterTuner(parser, save_path=tmp_path, auto_save_enabled=True, inactivity_timeout=2)

    def user(tuner, root):
        delays = [c.args[0] for c in root.after.call_args_list]
        assert 30000 in delays and 2000 in delays  # auto-save and inactivity timers
        tuner.entries["name"].set("idle-run")
        tuner._on_inactivity_timeout()
        assert root.after.call_args.args == (500, tuner._close)
        tuner._close()

    _run_gui(fake_tk, tuner, user)
    assert parser.parse_args([]).name == "idle-run"
    assert TIMESTAMPED.fullmatch(tuner.last_saved_path.name)


def test_gui_inactivity_can_be_disabled(fake_tk, tmp_path):
    tuner = ParameterTuner(
        make_parser(), save_path=tmp_path, auto_save_enabled=False, inactivity_timeout=None
    )

    def user(tuner, root):
        assert root.after.call_count == 0
        tuner._on_activity()
        assert root.after.call_count == 0
        tuner._auto_save_timer = None
        tuner.auto_save_var.set(True)
        tuner._toggle_auto_save()
        assert root.after.call_args.args == (tuner.save_delay, tuner._auto_save)
        tuner._close()

    _run_gui(fake_tk, tuner, user)


def test_gui_requires_display(fake_tk):
    fake_tk.Tk.side_effect = fake_tk.TclError("no display name and no $DISPLAY")
    with pytest.raises(RuntimeError, match="graphical display"):
        ParameterTuner(make_parser()).tune()
