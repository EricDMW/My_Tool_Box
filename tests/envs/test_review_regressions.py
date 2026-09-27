"""Regression tests for issues found in the 1.0 code review."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

import env_lib
from env_lib.utils import LIGHT, register_theme, set_theme


def test_torch_kuramoto_does_not_keep_autograd_graph():
    torch = pytest.importorskip("torch")
    env = env_lib.KuramotoOscillatorEnvTorch(n_oscillators=4)
    env.reset(seed=0)
    weights = torch.zeros(env.action_space.shape[0], requires_grad=True)
    for _ in range(5):
        env.step(weights + 0.1)
    assert not env.phases.requires_grad
    env.get_batch_observations().numpy()


def test_discrete_pistonball_rejects_non_integer_actions():
    pytest.importorskip("pymunk")
    env = env_lib.PistonballEnv(n_pistons=3, continuous=False)
    env.reset(seed=0)
    for bad in ([np.nan, 1, 1], [0.5, 1.0, 2.0]):
        with pytest.raises(ValueError):
            env.step(np.array(bad))
    env.step(np.array([0.0, 1.0, 2.0]))  # integral floats are fine


def test_pistonball_human_windows_are_independent():
    pytest.importorskip("pygame")
    pytest.importorskip("pymunk")
    a = env_lib.PistonballEnv(n_pistons=3, render_mode="human")
    b = env_lib.PistonballEnv(n_pistons=3, render_mode="human")
    for env in (a, b):
        env.reset(seed=0)
        env.step(env.action_space.sample())
    a.close()
    b.step(b.action_space.sample())
    b.close()


def test_pistonball_accepts_named_theme_colours():
    pytest.importorskip("pygame")
    pytest.importorskip("pymunk")
    register_theme(dataclasses.replace(LIGHT, name="named-colours", background="white"))
    set_theme("named-colours")
    try:
        env = env_lib.PistonballEnv(n_pistons=3, render_mode="rgb_array")
        env.reset(seed=0)
        assert env.render().shape[2] == 3
        env.close()
    finally:
        set_theme("dark")


def test_ajlatt_accepts_policy_instance_and_map_path():
    from env_lib.ajlatt_env.controllers import WaypointPolicy
    from env_lib.ajlatt_env.maps import MAP_DIR

    policy = WaypointPolicy([((-np.inf, np.inf), (30.0, 30.0))], velocity=0.2)
    env = env_lib.AJLATTEnv(map_name="obstacles02", target_policy=policy)
    assert env._target_policy is policy
    env = env_lib.AJLATTEnv(map_name=str(MAP_DIR / "obstacles04.yaml"))
    assert type(env._target_policy).__name__ == "WaypointPolicy"


def test_map_named_empty_keeps_its_obstacles(tmp_path):
    from env_lib.ajlatt_env.maps import load_grid_map
    from env_lib.ajlatt_env.maps.builder import build_map

    spec = {
        "map_info": {"name": "my_empty_room", "width": 10.0, "height": 10.0, "resolution": 0.5},
        "obstacles": [{"type": "circle", "center": [5.0, 5.0], "radius": 1.0}],
    }
    build_map(spec, tmp_path)
    grid = load_grid_map(tmp_path / "my_empty_room")
    assert grid.map is not None and grid.map.sum() > 0
    assert load_grid_map("empty").map is None


def test_legacy_ajlatt_imports():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from env_lib.ajlatt_env import ajlatt_env
        from env_lib.ajlatt_env.env import make

        assert isinstance(ajlatt_env(map_name="empty"), env_lib.AJLATTEnv)
        assert isinstance(make(map_name="empty"), env_lib.AJLATTEnv)
        env = env_lib.make("AJLATT-v0", render=False)
    assert env.unwrapped.render_mode is None


def test_plot_map_samples_dynamic_maps():
    from env_lib.ajlatt_env.maps.plotting import plot_map

    ax = plot_map("dynamic_map")
    assert ax.images[0].get_array().sum() > 0


def test_ajlatt_render_before_reset_raises():
    env = env_lib.AJLATTEnv(render_mode="rgb_array")
    with pytest.raises(RuntimeError, match="reset"):
        env.render()


def test_info_arrays_are_not_aliased():
    env = env_lib.AJLATTEnv()
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros((4, 2)))
    assert info["agent_rewards"] is not reward
    wireless = env_lib.WirelessCommEnv(grid_x=3, grid_y=3)
    wireless.reset(seed=0)
    _, _, _, _, info = wireless.step(wireless.action_space.sample())
    info["ap_load"][:] = 99
    assert not np.any(wireless.last_ap_load == 99)


def test_wireless_rejects_fractional_actions():
    env = env_lib.WirelessCommEnv(grid_x=2, grid_y=2)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step([0.9, 0.9, 0.9, 4.9])
    env.step([0.0, 1.0, 2.0, 4.0])


def test_linemsg_two_agents():
    env = env_lib.LineMsgEnv(num_agents=2)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2, 3)
    env.step(3)
