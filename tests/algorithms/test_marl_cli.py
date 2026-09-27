"""Tests for the marl-train command line and the presets module."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from marl_algorithms.__main__ import main
from marl_algorithms.core.base import Algorithm
from marl_algorithms.presets import get_preset, list_presets, train_preset
from marl_algorithms.registry import list_algorithms


def test_list_and_presets_commands(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for info in list_algorithms():
        assert info.name in out and info.reference in out
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    assert "mappo" in out and "PowerGrid-v0" in out


def test_every_algorithm_has_a_preset_and_presets_are_well_formed():
    presets = list_presets()
    assert {name for name, _ in presets} == {info.name for info in list_algorithms()}
    for name, env_id in presets:
        preset = get_preset(name, env_id)
        assert int(preset["total_steps"]) > 0 and int(preset["num_envs"]) > 0
        assert isinstance(preset.get("config", {}), dict)
    assert get_preset("mappo", "NoSuchEnv-v0") is None
    with pytest.raises(KeyError):
        get_preset("reinforce", "PowerGrid-v0")


def test_get_preset_returns_a_copy():
    preset = get_preset("qmix", "LineMsg-v0")
    preset["config"]["lr"] = -1.0
    assert get_preset("qmix", "LineMsg-v0")["config"].get("lr") != -1.0


def test_train_preset_with_overrides():
    algo, log = train_preset("vdn", "LineMsg-v0", total_steps=600, num_envs=4, warmup_steps=200)
    assert algo.env_steps >= 600 and log.env_id == "LineMsg-v0"
    with pytest.raises(KeyError):
        train_preset("maddpg", "LineMsg-v0")
    small, _ = train_preset(
        "vdn", "LineMsg-v0", total_steps=200, num_envs=4, env_kwargs={"num_agents": 4}
    )
    assert small.spec.n_agents == 4


def test_run_and_evaluate_commands(tmp_path, capsys):
    checkpoint = tmp_path / "mappo.pt"
    csv = tmp_path / "mappo.csv"
    code = main(
        [
            "run",
            "mappo",
            "Consensus-v0",
            "--no-preset",
            "--steps",
            "512",
            "--num-envs",
            "4",
            "--set",
            "rollout_length=32",
            "hidden_sizes=(16,)",
            "--env-kwarg",
            "n_agents=3",
            "max_steps=20",
            "--eval-episodes",
            "4",
            "--save",
            str(checkpoint),
            "--csv",
            str(csv),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Training mappo on Consensus-v0 (defaults)" in out
    assert "random" in out and "baseline" in out
    assert checkpoint.exists() and csv.read_text().startswith("env_steps,return,length")
    assert Algorithm.load(checkpoint).name == "mappo"
    assert (
        main(
            [
                "evaluate",
                str(checkpoint),
                "Consensus-v0",
                "--episodes",
                "4",
                "--env-kwarg",
                "n_agents=3",
                "max_steps=20",
            ]
        )
        == 0
    )
    assert "mappo" in capsys.readouterr().out


def test_bad_key_value_pair_exits():
    with pytest.raises(SystemExit):
        main(["run", "vdn", "LineMsg-v0", "--set", "lr"])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "reinforce", "LineMsg-v0"], "unknown algorithm 'reinforce'"),
        (["run", "vdn", "NoSuchEnv-v0"], "unknown environment 'NoSuchEnv-v0'"),
        (["run", "vdn", "LineMsg-v0", "--set", "learning_rate=0.1"], "learning_rate"),
        (["evaluate", "missing.pt", "NoSuchEnv-v0"], "unknown environment"),
        (["evaluate", "missing.pt", "LineMsg-v0"], "no checkpoint"),
    ],
)
def test_invalid_arguments_give_a_clean_error(argv, message, capsys):
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("marl-train: error:") and message in captured.err


def test_preset_helpers_are_exported():
    import marl_algorithms

    assert marl_algorithms.train_preset is train_preset
    assert marl_algorithms.get_preset is get_preset
    assert marl_algorithms.list_presets is list_presets
