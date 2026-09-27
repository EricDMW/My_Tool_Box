"""Regression tests for issues found in the 1.2.1 pre-submission review."""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pytest

import env_lib
from env_lib.__main__ import main
from env_lib.wrappers import to_parallel

_NEEDS = {"Pistonball-v0": ("pymunk", "pygame"), "KuramotoOscillatorTorch": ("torch",)}


def _skip_missing(env_id: str) -> None:
    for prefix, modules in _NEEDS.items():
        if env_id.startswith(prefix):
            for module in modules:
                pytest.importorskip(module)


def _make(env_id: str, **kwargs):
    _skip_missing(env_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return env_lib.make(env_id, **kwargs)


# ---------------------------------------------------------------------------
# Pickling and copying
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_id", env_lib.list_envs())
def test_environments_pickle_and_copy_after_rendering(env_id):
    env = _make(env_id, render_mode="rgb_array")
    env.reset(seed=3)
    env.action_space.seed(0)
    env.step(env.action_space.sample())
    env.render()
    for clone in (pickle.loads(pickle.dumps(env)), copy.deepcopy(env)):
        action = env.action_space.sample()
        expected = env.unwrapped.__class__  # same class after the round trip
        assert isinstance(clone.unwrapped, expected)
        obs_a, reward_a, *_ = copy.deepcopy(env).step(action)
        obs_b, reward_b, *_ = clone.step(action)
        np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))
        np.testing.assert_array_equal(np.asarray(reward_a), np.asarray(reward_b))
        frame = clone.render()  # the renderer is rebuilt
        assert frame.ndim == 3 and frame.dtype == np.uint8
        clone.close()
    env.close()


@pytest.mark.parametrize(
    "env_id", ["PowerGrid-v0", "Platoon-v0", "Consensus-v0", "KuramotoOscillator-v0"]
)
def test_native_vector_environments_pickle_after_rendering(env_id):
    envs = env_lib.make_vec(env_id, 3, render_mode="rgb_array")
    envs.reset(seed=0)
    envs.render()
    clone = pickle.loads(pickle.dumps(envs))
    actions = envs.action_space.sample()
    np.testing.assert_array_equal(envs.step(actions)[0], clone.step(actions)[0])
    clone.close()
    envs.close()


def test_failed_construction_prints_no_del_errors():
    code = (
        "import env_lib\n"
        "for env_id in ('PowerGrid-v0', 'Platoon-v0', 'Consensus-v0', 'KuramotoOscillator-v0'):\n"
        "    try:\n"
        "        env_lib.make_vec(env_id, 2, max_episode_steps=-5)\n"
        "    except (TypeError, ValueError):\n"
        "        pass\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert "Exception ignored" not in result.stderr


# ---------------------------------------------------------------------------
# make / make_vec arguments and seeding
# ---------------------------------------------------------------------------
def test_unknown_keyword_argument_is_reported_clearly():
    with pytest.raises(TypeError, match=r"PowerGrid-v0 does not accept the argument 'n_bus'"):
        env_lib.make("PowerGrid-v0", n_bus=4)
    with pytest.raises(TypeError, match=r"env-lib describe PowerGrid-v0"):
        env_lib.make_vec("PowerGrid-v0", 2, n_bus=4)


@pytest.mark.parametrize("num_envs", [0, -1, 2.5, True])
def test_make_vec_rejects_invalid_num_envs(num_envs):
    with pytest.raises(ValueError, match="num_envs must be a positive integer"):
        env_lib.make_vec("Formation-v0", num_envs)


def test_vector_reset_accepts_a_list_of_seeds():
    envs = env_lib.make_vec("Formation-v0", 3)
    first, _ = envs.reset(seed=[4, 5, 6])
    again, _ = envs.reset(seed=[4, 5, 6])
    other, _ = envs.reset(seed=[4, 5, 7])
    # The batch shares one generator: the list seeds it as a whole.
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    with pytest.raises(ValueError):
        envs.reset(seed=[1, 2])
    envs.close()


def test_to_parallel_with_several_kuramoto_systems():
    env = _make("KuramotoOscillatorTorch-v1")
    parallel = to_parallel(env)
    observations, _ = parallel.reset(seed=0)
    actions = {agent: parallel.action_space(agent).sample() for agent in parallel.agents}
    _, rewards, *_ = parallel.step(actions)
    assert set(rewards) == set(observations)
    assert all(np.isfinite(value) for value in rewards.values())
    parallel.close()


# ---------------------------------------------------------------------------
# Imports without optional extras
# ---------------------------------------------------------------------------
def test_star_imports_leave_out_names_that_need_an_extra():
    for module in ("env_lib", "env_lib.kos_env", "toolkit"):
        names = __import__(module, fromlist=["__all__"]).__all__
        assert not {"KuramotoOscillatorEnvTorch", "PistonballEnv", "pistonball_env"} & set(names)
        assert "neural_toolkit" not in names
    namespace: dict = {}
    exec("from env_lib import *", namespace)
    assert "make" in namespace and "PowerGridEnv" in namespace
    assert "KuramotoOscillatorEnvTorch" in dir(env_lib)
    assert "neural_toolkit" in dir(__import__("toolkit"))


def test_creating_an_environment_does_not_import_matplotlib():
    code = (
        "import sys, env_lib\n"
        "env = env_lib.make('PowerGrid-v0'); env.reset(seed=0)\n"
        "envs = env_lib.make_vec('Consensus-v0', 2); envs.reset(seed=0)\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("command", ["run", "evaluate", "bench"])
def test_cli_reports_invalid_values_without_traceback(command, capsys):
    args = [command, "PowerGrid-v0", "--kwarg", "n_buses=1"]
    if command == "bench":
        args += ["--steps", "2"]
    assert main(args) == 2
    assert "env-lib: error:" in capsys.readouterr().err


def test_cli_checks_the_animation_format_before_the_rollout(tmp_path, capsys):
    assert main(["run", "Formation-v0", "--gif", str(tmp_path / "out.webp")]) == 2
    err = capsys.readouterr().err
    assert "unsupported animation format" in err
    assert not (tmp_path / "out.webp").exists()


def test_describe_warns_about_reward_design():
    assert "caution" in env_lib.describe("KuramotoOscillator-v0")
    assert "terminate_on_collision=False" in env_lib.describe("AJLATT-v0")
    assert "caution" not in env_lib.describe("KuramotoOscillator-FreqSync-Constant-v0")
    assert "caution" not in env_lib.describe("PowerGrid-v0")
