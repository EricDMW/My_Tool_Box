"""Regression tests for marl_algorithms issues found in the 1.2.1 pre-submission review."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import env_lib
import marl_algorithms
from marl_algorithms import compare, make_algorithm, torch_threads, train
from marl_algorithms.__main__ import main
from marl_algorithms.core import make_vector_env
from marl_algorithms.core.base import Algorithm, OnPolicyAlgorithm


def test_torch_threads_sets_and_restores_the_thread_count():
    before = torch.get_num_threads()
    with torch_threads(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == before
    with torch_threads(None):
        assert torch.get_num_threads() == before
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError), torch_threads(bad):
            pass


def test_train_uses_one_thread_and_restores_the_setting(monkeypatch):
    seen = []
    original = OnPolicyAlgorithm.learn

    def learn(self, *args, **kwargs):
        seen.append(torch.get_num_threads())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(OnPolicyAlgorithm, "learn", learn)
    before = torch.get_num_threads()
    algo, _ = train("ippo", "Formation-v0", 64, num_envs=2, rollout_length=32)
    assert seen == [1]
    assert torch.get_num_threads() == before
    assert algo.metadata == {"env_id": "Formation-v0", "env_kwargs": {}}


def test_unknown_algorithm_is_reported_before_creating_environments(monkeypatch):
    def fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("environment created")

    monkeypatch.setattr("marl_algorithms.registry.make_vector_env", fail)
    with pytest.raises(KeyError):
        train("mapo", "Formation-v0", 64)


def test_budget_and_device_validation():
    envs = make_vector_env("Formation-v0", 2)
    algo = make_algorithm("ippo", envs)
    for bad in (0, -5, 2.5):
        with pytest.raises(ValueError, match="total_steps"):
            algo.learn(envs, bad)
    envs.close()
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="cuda"):
            make_algorithm("ippo", env_lib.make("Formation-v0"), device="cuda")


def test_off_policy_warns_when_the_budget_ends_during_warmup():
    envs = make_vector_env("Formation-v0", 2)
    algo = make_algorithm("maddpg", envs)
    with pytest.warns(UserWarning, match="warmup"):
        algo.learn(envs, algo.config.warmup_steps // 2)
    assert algo.num_updates == 0
    envs.close()


def test_check_env_reports_a_different_environment():
    algo = make_algorithm("ippo", env_lib.make("Formation-v0"))
    algo.check_env(env_lib.make_vec("Formation-v0", 2))  # same structure
    with pytest.raises(ValueError, match="was built for"):
        algo.evaluate(env_lib.make_vec("Formation-v0", 2, n_agents=6), n_episodes=2)
    with pytest.raises(ValueError, match="was built for"):
        algo.check_env(env_lib.make("PowerGrid-v0"))


def test_checkpoint_round_trip_and_load_errors(tmp_path):
    algo, _ = train("ippo", "Formation-v0", 64, num_envs=2, rollout_length=32)
    path = algo.save(tmp_path / "ippo.pt")
    loaded = Algorithm.load(path)
    assert loaded.metadata == algo.metadata
    obs = np.zeros((2, *algo.spec.obs_shape), dtype=np.float32)
    np.testing.assert_allclose(
        loaded.act(obs, deterministic=True), algo.act(obs, deterministic=True)
    )

    with pytest.raises(FileNotFoundError):
        Algorithm.load(tmp_path / "missing.pt")
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="not a marl_algorithms checkpoint"):
        Algorithm.load(junk)
    other = tmp_path / "other.pt"
    torch.save({"weights": 1}, other)
    with pytest.raises(ValueError, match="not a marl_algorithms checkpoint"):
        Algorithm.load(other)


def test_compare_without_presets_needs_explicit_algorithms():
    with pytest.raises(ValueError, match="algorithms="):
        compare("Formation-v0")
    report = compare("Formation-v0", [], n_episodes=4)
    assert [row.name for row in report.rows] == ["random", "baseline"]


def test_cli_checks_output_paths_before_training(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("")
    code = main(
        ["run", "ippo", "Formation-v0", "--steps", "64", "--save", str(blocker / "model.pt")]
    )
    assert code == 2
    assert "is not a directory" in capsys.readouterr().err
    assert main(["run", "ippo", "Formation-v0", "--steps", "64", "--csv", str(tmp_path)]) == 2
    assert "is a directory" in capsys.readouterr().err


def test_cli_evaluate_notes_a_different_environment(tmp_path, capsys):
    algo, _ = train("ippo", "Formation-v0", 64, num_envs=2, rollout_length=32)
    path = algo.save(tmp_path / "ippo.pt")
    assert main(["evaluate", str(path), "Formation-v0", "--episodes", "2"]) == 0
    assert "note:" not in capsys.readouterr().err
    assert main(["evaluate", str(path), "Consensus-v0", "--episodes", "2"]) == 0
    assert "trained on Formation-v0" in capsys.readouterr().err
    assert main(["evaluate", str(path), "PowerGrid-v0", "--episodes", "2"]) == 2
    assert "was built for" in capsys.readouterr().err
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"x")
    assert main(["evaluate", str(junk), "Formation-v0"]) == 2
    assert "not a marl_algorithms checkpoint" in capsys.readouterr().err


def test_package_exports_new_names():
    assert marl_algorithms.torch_threads is torch_threads
    assert marl_algorithms.ComparisonRow.__name__ == "ComparisonRow"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        namespace: dict = {}
        exec("from marl_algorithms import *", namespace)
    assert "compare" in namespace and "torch_threads" in namespace
