"""Tests for the environment registry and the lazy top-level API."""

import gymnasium as gym
import pytest

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


@pytest.mark.parametrize(
    ("env_id", "attribute"),
    [
        ("KuramotoOscillator-v0", "max_steps"),
        ("LineMsg-v0", "max_iter"),
        ("WirelessComm-v1", "max_iter"),
        ("Consensus-v0", "max_steps"),
    ],
)
def test_max_episode_steps_sets_the_environment_limit(env_id, attribute):
    env = env_lib.make(env_id, max_episode_steps=7)
    assert getattr(env.unwrapped, attribute) == 7
    # No TimeLimit wrapper is stacked on top of the environment's own limit.
    wrapper = env
    while hasattr(wrapper, "env"):
        assert type(wrapper).__name__ != "TimeLimit"
        wrapper = wrapper.env
    env.close()


def test_max_episode_steps_reaches_ajlatt_config():
    env = env_lib.make("AJLATT-v0", max_episode_steps=9)
    assert env.unwrapped.config.max_episode_steps == 9
    env.close()


def test_max_episode_steps_conflict_raises():
    with pytest.raises(ValueError):
        env_lib.make("LineMsg-v0", max_episode_steps=5, max_iter=6)


def test_every_spec_has_a_valid_limit_argument():
    import inspect

    import gymnasium as gym

    for spec in env_lib.registration.ENV_SPECS:
        module_name, class_name = spec.entry_point.split(":")
        try:
            cls = getattr(__import__(module_name, fromlist=[class_name]), class_name)
        except ImportError:
            continue
        params = inspect.signature(cls.__init__).parameters
        accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
        assert spec.limit_kwarg in params or accepts_kwargs, spec.id
        assert gym.spec(spec.id).vector_entry_point == spec.vector_entry_point


@pytest.mark.parametrize("mode", [None, "sync"])
@pytest.mark.parametrize("autoreset_mode", ["same_step", "SameStep"])
def test_make_vec_autoreset_mode_for_every_vectorization_mode(mode, autoreset_mode):
    envs = env_lib.make_vec(
        "Consensus-v0",
        2,
        vectorization_mode=mode,
        autoreset_mode=autoreset_mode,
        max_episode_steps=1,
    )
    envs.reset(seed=0)
    _, _, _, truncated, infos = envs.step(envs.action_space.sample())
    assert truncated.all()
    assert "final_obs" in infos and infos["_final_obs"].all()
    envs.close()


def test_make_vec_rejects_unknown_autoreset_mode():
    with pytest.raises(ValueError):
        env_lib.make_vec("LineMsg-v0", 2, autoreset_mode="sometimes")


def test_make_vec_reduces_per_agent_rewards_for_sync_vectors():
    envs = env_lib.make_vec("AJLATT-v0", 2, max_episode_steps=3)
    envs.reset(seed=0)
    _, rewards, terminated, _, infos = envs.step(envs.action_space.sample())
    assert rewards.shape == (2,) and terminated.shape == (2,)
    assert infos["agent_rewards"].shape == (2, 4)
    envs.close()
