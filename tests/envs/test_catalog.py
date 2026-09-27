"""Tests for env_lib.catalog (catalog, describe, EnvInfo)."""

from __future__ import annotations

import dataclasses
import sys

import pytest

import env_lib
from env_lib import registration
from env_lib.catalog import Catalog, EnvInfo, catalog, clear_cache, describe
from env_lib.registration import EnvSpec


def test_catalog_lists_every_registered_environment():
    entries = catalog()
    assert isinstance(entries, Catalog) and all(isinstance(e, EnvInfo) for e in entries)
    assert entries.ids() == env_lib.list_envs()


def test_inspected_fields():
    entries = catalog(family="consensus")
    info = entries.get("Consensus-v0")
    assert info.family == "consensus" and info.available and info.native_vector
    assert info.n_agents == 8
    assert info.observation_shape == (8, 16) and info.action_shape == (8, 2)
    assert info.kwargs == {}
    assert entries.get("Formation-v0").kwargs == {"task": "formation"}
    assert info.observation_space.startswith("Box(8, 16) float32")
    kuramoto = catalog(family="kuramoto").get("KuramotoOscillator-Constant-v0")
    assert kuramoto.n_agents == 1 and kuramoto.observation_shape == (18,)
    line = catalog(family="linemsg").get("LineMsg-v0")
    assert line.action_shape == () and line.action_space.startswith("Discrete(1024)")
    with pytest.raises(KeyError):
        entries.get("LineMsg-v0")


def test_entries_are_immutable():
    info = catalog(family="consensus", inspect=False)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.n_agents = 3
    with pytest.raises(TypeError):
        info.kwargs["task"] = "consensus"


def test_filters():
    continuous = catalog(action_type="continuous", inspect=False).ids()
    discrete = catalog(action_type="discrete", inspect=False).ids()
    assert "Pistonball-v0" in continuous and "Pistonball-v0" in discrete  # continuous|discrete
    assert "Consensus-v0" in continuous and "Consensus-v0" not in discrete
    assert "LineMsg-v0" in discrete and "LineMsg-v0" not in continuous
    assert catalog(observation_type="discrete", inspect=False).ids() == [
        "LineMsg-v0",
        "WirelessComm-v0",
        "WirelessComm-v1",
    ]
    native = catalog(native_vector=True, inspect=False)
    assert native and all(info.native_vector for info in native)
    assert "AJLATT-v0" in catalog(native_vector=False, inspect=False).ids()
    both = catalog(family=("linemsg", "ajlatt"), inspect=False).ids()
    assert both == ["LineMsg-v0", "AJLATT-v0"]
    with pytest.raises(ValueError):
        catalog(action_type="hybrid")


def test_without_inspection_no_shapes():
    info = catalog(family="consensus", inspect=False)[0]
    assert info.n_agents is None and info.observation_shape is None
    assert info.available


def test_text_and_markdown_tables():
    entries = catalog(family=("consensus", "linemsg"))
    text = str(entries)
    lines = text.splitlines()
    assert lines[0].startswith("ID") and "AGENTS" in lines[0]
    assert len(lines) == 1 + len(entries)
    assert any(line.startswith("Formation-v0") and "(8, 16)" in line for line in lines)
    assert text.isascii()
    markdown = entries.to_markdown().splitlines()
    assert markdown[0].startswith("| Id |") and markdown[1].startswith("|---")
    assert any("`LineMsg-v0`" in line and "Discrete(1024)" in line for line in markdown[2:])
    assert len(markdown) == 2 + len(entries)
    assert str(Catalog()) == "(no matching environments)"


def test_unavailable_environments_do_not_crash(monkeypatch):
    fake = [
        EnvSpec("Missing-v0", "env_lib.no_such_env:NoEnv", family="missing"),
        EnvSpec(
            "NeedsExtra-v0",
            "env_lib.consensus_env.consensus_env:ConsensusEnv",
            family="consensus",
            requires="nonexistent_extra_dependency",
        ),
        EnvSpec(
            "Broken-v0",
            "env_lib.consensus_env.consensus_env:ConsensusEnv",
            {"n_agents": -3},
            family="consensus",
        ),
    ]
    catalog_module = sys.modules["env_lib.catalog"]
    specs = [*registration.ENV_SPECS, *fake]
    monkeypatch.setattr(catalog_module, "ENV_SPECS", specs)
    monkeypatch.setattr(registration, "ENV_SPECS", specs)
    clear_cache()
    try:
        entries = catalog()
        missing = entries.get("Missing-v0")
        assert not missing.available and "no_such_env" in missing.note
        extra = entries.get("NeedsExtra-v0")
        assert not extra.available and "my-tool-box[nonexistent_extra_dependency]" in extra.note
        broken = entries.get("Broken-v0")
        assert not broken.available and "could not be created" in broken.note
        assert "Broken-v0" not in catalog(available_only=True).ids()
        assert "unavailable" in str(entries)
        text = describe("Missing-v0")
        assert "unavailable" in text and "no_such_env" in text
    finally:
        clear_cache()


def test_describe_contents():
    text = describe("Formation-v0")
    assert text.startswith("Formation-v0 -- networked formation control")
    for label in ("summary", "observation", "action", "obs layout", "parameters", "vector"):
        assert f"\n{label}" in text
    assert "task='formation'*" in text and "(* = set by this id)" in text
    assert "position" in text and "neighbors" in text
    assert "Laplacian" in text and "no optional dependencies" in text
    assert text.isascii()
    ajlatt = describe("AJLATT-v0")
    assert "self_pose" in ajlatt and "map_name='obstacles04'" in ajlatt
    kuramoto = describe("KuramotoOscillator-v0")
    assert "high [1 x10, 5 x45]" in kuramoto
    torch_text = describe("KuramotoOscillatorTorch-v0")
    assert 'extra "torch"' in torch_text
    with pytest.raises(KeyError):
        describe("NoSuchEnv-v0")


def test_module_is_callable():
    module = sys.modules["env_lib.catalog"]
    assert module(family="ajlatt", inspect=False).ids() == ["AJLATT-v0"]
    assert env_lib.catalog(family="ajlatt", inspect=False).ids() == ["AJLATT-v0"]
