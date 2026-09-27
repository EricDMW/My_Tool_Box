"""Tests for the env-lib command line (env_lib.__main__.main)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from env_lib.__main__ import main


def test_list(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("ID")
    assert "Consensus-v0" in out and "AJLATT-v0" in out and "environment(s)" in out
    assert out.isascii()


def test_list_filters_and_markdown(capsys):
    assert main(["list", "--continuous", "--no-inspect"]) == 0
    out = capsys.readouterr().out
    assert "Pistonball-v0" in out and "LineMsg-v0" not in out
    assert main(["list", "--discrete", "--markdown", "--no-inspect"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| Id |") and "`WirelessComm-v1`" in out and "Consensus" not in out
    assert main(["list", "--family", "consensus", "--native", "--no-inspect"]) == 0
    assert "Formation-v0" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["list", "--continuous", "--discrete"])


def test_describe_and_unknown_id(capsys):
    assert main(["describe", "Consensus-v0"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Consensus-v0 --") and "obs layout" in out
    assert main(["describe", "Consens-v0"]) == 2
    err = capsys.readouterr().err
    assert "unknown environment" in err and "Consensus-v0" in err


def test_baselines(capsys):
    assert main(["baselines"]) == 0
    out = capsys.readouterr().out
    assert "consensus" in out and "wireless_comm" in out


def test_run_baseline(capsys):
    assert main(["run", "Consensus-v0", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "policy        baseline" in out and "terminated" in out
    assert "team return" in out and "agent_7" in out and "success=True" in out


def test_run_random_with_kwargs(capsys):
    code = main(
        ["run", "LineMsg-v0", "--policy", "random", "--steps", "5", "--kwarg", "num_agents=4"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "LineMsg-v0 (num_agents=4)" in out and "5 steps, stopped (--steps)" in out
    assert "agent_3" in out and "agent_4" not in out
    assert main(["run", "LineMsg-v0", "--kwarg", "num_agents"]) == 2
    assert "key=value" in capsys.readouterr().err
    assert main(["run", "LineMsg-v0", "--kwarg", "bogus=1"]) == 2
    assert "invalid arguments" in capsys.readouterr().err


def test_run_records_gif(tmp_path, capsys):
    path = tmp_path / "formation.gif"
    code = main(["run", "Formation-v0", "--steps", "4", "--gif", str(path), "--theme", "light"])
    assert code == 0
    assert path.exists() and path.stat().st_size > 0
    out = capsys.readouterr().out
    assert f"{path} (5 frames)" in out
    from env_lib.utils.rendering import set_theme

    set_theme("dark")  # restore the default for other tests


def test_evaluate(capsys):
    assert main(["evaluate", "LineMsg-v0", "--episodes", "3"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[1].split()[:2] == ["policy", "mean"]
    assert lines[2].startswith("random") and lines[3].startswith("baseline")
    assert main(["evaluate", "Consensus-v0", "--episodes", "4", "--num-envs", "2"]) == 0
    assert "2 parallel copies" in capsys.readouterr().out


def test_bench(capsys):
    assert main(["bench", "Consensus-v0", "--num-envs", "4", "--steps", "5"]) == 0
    out = capsys.readouterr().out
    assert "single" in out and "sync" in out and "env steps/s" in out
    assert main(["bench", "LineMsg-v0", "--num-envs", "2", "--steps", "3"]) == 0
    assert "no native vector implementation" in capsys.readouterr().out


def test_python_dash_m():
    result = subprocess.run(
        [sys.executable, "-m", "env_lib", "list", "--no-inspect", "--family", "ajlatt"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "AJLATT-v0" in result.stdout
