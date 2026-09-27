"""Tests for marl_algorithms.baselines: compare(), per_copy() and marl-train compare."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

import env_lib
from marl_algorithms import Comparison, compare, make_vector_env, per_copy, train
from marl_algorithms.__main__ import main
from marl_algorithms.baselines import ComparisonRow

QUICK = {"total_steps": 400, "num_envs": 4, "n_episodes": 4, "warmup_steps": 100}


def always_relay(obs: np.ndarray) -> np.ndarray:
    """LineMsg: every agent relays (the optimal policy), for batched observations."""
    return np.ones(obs.shape[:-1], dtype=np.int8)


def test_compare_rows_order_and_statistics(tmp_path):
    trained, _ = train("vdn", "LineMsg-v0", 400, num_envs=4, seed=0, warmup_steps=100)
    report = compare(
        "LineMsg-v0",
        ["iql", "vdn"],
        seeds=(0, 1),
        policies={"relay": always_relay, "my vdn": trained},
        **QUICK,
    )
    assert isinstance(report, Comparison)
    assert [row.name for row in report.rows] == [
        "random",
        "baseline",
        "iql",
        "vdn",
        "relay",
        "my vdn",
    ]
    assert [row.kind for row in report.rows] == ["reference"] * 2 + ["algorithm"] * 2 + [
        "policy"
    ] * 2
    iql = report["iql"]
    assert iql.seeds == (0, 1) and len(iql.returns) == 2 and len(iql.env_steps) == 2
    assert iql.mean == pytest.approx(np.mean(iql.returns))
    assert set(report.algorithms) == {("iql", 0), ("iql", 1), ("vdn", 0), ("vdn", 1)}
    assert set(report.logs) == set(report.algorithms)
    # The optimal policy and the classical controller agree on LineMsg.
    assert report["relay"].mean == pytest.approx(report["baseline"].mean)
    assert math.isnan(report["relay"].std)
    assert report.ranking()[0].name in {"relay", "iql", "vdn", "my vdn"}
    assert {row.kind for row in report.ranking()} <= {"algorithm", "policy"}
    with pytest.raises(KeyError):
        report["missing"]

    text = str(report)
    assert "LineMsg-v0" in text and "relay" in text and "baseline" in text
    assert report.to_markdown().startswith("| Method | Kind |")
    records = report.records()
    assert records[2]["seeds"] == [0, 1] and records[0]["env_steps"] is None
    path = report.to_csv(tmp_path / "report.csv")
    lines = path.read_text().splitlines()
    assert lines[0].startswith("env_id,name,kind,mean_return") and len(lines) == 7


def test_compare_is_reproducible_and_uses_shared_episodes():
    first = compare("LineMsg-v0", ["iql"], **QUICK)
    second = compare("LineMsg-v0", ["iql"], **QUICK)
    assert first["iql"].returns == second["iql"].returns
    assert first["random"].returns == second["random"].returns


def test_compare_defaults_to_the_preset_algorithms():
    report = compare(
        "LineMsg-v0",
        include_random=False,
        include_baseline=False,
        total_steps=400,
        num_envs=4,
        n_episodes=4,
    )
    assert [row.name for row in report.rows] == ["ippo", "mappo", "iql", "vdn", "qmix"]


def test_compare_without_algorithms_and_with_env_kwargs():
    report = compare(
        "LineMsg-v0",
        [],
        policies={"relay": always_relay},
        n_episodes=4,
        env_kwargs={"num_agents": 4},
    )
    assert [row.name for row in report.rows] == ["random", "baseline", "relay"]
    assert report.env_kwargs == {"num_agents": 4}


def test_compare_skips_the_controller_where_there_is_none(monkeypatch):
    import env_lib.baselines as baselines

    def no_controller(env, **kwargs):
        raise NotImplementedError

    monkeypatch.setattr(baselines, "baseline_policy", no_controller)
    report = compare("LineMsg-v0", [], n_episodes=2)
    assert [row.name for row in report.rows] == ["random"]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"algorithms": ["reinforce"]}, KeyError),
        ({"algorithms": ["maddpg"]}, ValueError),  # no preset and no total_steps
        ({"algorithms": [], "seeds": ()}, ValueError),
        ({"algorithms": [], "n_episodes": 0}, ValueError),
        ({"algorithms": ["iql"], "policies": {"iql": always_relay}}, ValueError),
        ({"algorithms": [], "policies": {"random": always_relay}}, ValueError),
    ],
)
def test_compare_rejects_invalid_arguments(kwargs, error):
    with pytest.raises(error):
        compare("LineMsg-v0", **kwargs)


def test_per_copy_batches_a_single_environment_policy():
    envs = make_vector_env("LineMsg-v0", 3)
    obs, _ = envs.reset(seed=0)
    single = lambda o: np.ones(o.shape[0], dtype=np.int8)  # noqa: E731
    actions = per_copy(single)(obs)
    assert actions.shape == (3, obs.shape[1])
    result = env_lib.evaluate(envs, per_copy(single), n_episodes=3, seed=1)
    assert result.mean_return == pytest.approx(
        env_lib.evaluate(envs, always_relay, n_episodes=3, seed=1).mean_return
    )
    envs.close()


def test_comparison_row_statistics():
    row = ComparisonRow("x", "algorithm", (1.0, 3.0), seeds=(0, 1))
    assert row.mean == 2.0 and row.std == pytest.approx(math.sqrt(2.0))


def test_compare_command(tmp_path, capsys):
    csv = tmp_path / "compare.csv"
    save_dir = tmp_path / "runs"
    code = main(
        [
            "compare",
            "LineMsg-v0",
            "--algos",
            "iql",
            "--seeds",
            "0",
            "1",
            "--steps",
            "400",
            "--num-envs",
            "4",
            "--episodes",
            "4",
            "--set",
            "warmup_steps=100",
            "--csv",
            str(csv),
            "--save-dir",
            str(save_dir),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Comparing iql on LineMsg-v0 (2 seeds)" in out and "baseline" in out
    assert csv.exists() and sorted(p.name for p in save_dir.iterdir()) == [
        "iql_seed0.pt",
        "iql_seed1.pt",
    ]
    assert (
        main(
            [
                "compare",
                "LineMsg-v0",
                "--algos",
                "iql",
                "--steps",
                "400",
                "--markdown",
                "--episodes",
                "2",
                "--num-envs",
                "4",
                "--set",
                "warmup_steps=100",
            ]
        )
        == 0
    )
    assert "| Method | Kind |" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["compare", "NoSuchEnv-v0"], "unknown environment"),
        (["compare", "LineMsg-v0", "--algos", "reinforce"], "unknown algorithm"),
        (["compare", "LineMsg-v0", "--algos", "maddpg"], "does not support the discrete"),
        (["compare", "PowerGrid-v0", "--algos", "iql"], "does not support the continuous"),
        (["compare", "Formation-v0", "--algos", "mappo"], "no preset"),
        (["compare", "LineMsg-v0", "--algos", "iql", "--set", "learning_rate=1"], "learning_rate"),
    ],
)
def test_compare_command_errors(argv, message, capsys):
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("marl-train: error:") and message in captured.err
