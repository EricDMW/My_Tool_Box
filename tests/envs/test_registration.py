"""Tests for the environment registry and the lazy top-level API."""

import gymnasium as gym

import env_lib


def test_all_ids_registered():
    ids = env_lib.list_envs()
    assert len(ids) == len(set(ids))
    for env_id in ids:
        assert env_id in gym.registry


def test_registrations_do_not_force_time_limits():
    for env_id in env_lib.list_envs():
        assert gym.registry[env_id].max_episode_steps is None


def test_unknown_attribute_raises():
    try:
        env_lib.does_not_exist
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError")


def test_make_hides_misleading_version_warning():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = env_lib.make("KuramotoOscillator-v0")
        env.close()
    assert not any("out of date" in str(w.message) for w in caught)
