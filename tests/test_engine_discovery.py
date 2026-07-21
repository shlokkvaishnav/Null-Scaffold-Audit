"""Unit tests for engine.discovery.discover_plugins.

These monkeypatch importlib.metadata.entry_points directly, so they require
no package to actually be pip-installed -- unlike tests/test_cli.py, which
depends on this repo's own "sde.plugins" entry points being real (i.e. the
package pip-installed), these test the discovery mechanism itself in
isolation.
"""

from __future__ import annotations

import pytest

from engine.discovery import discover_plugins
from engine.registry import PluginRegistry


class _FakeEntryPoint:
    def __init__(self, name, register_fn):
        self.name = name
        self._register_fn = register_fn

    def load(self):
        return self._register_fn


def _fake_entry_points_returning(eps):
    def _fake(*, group):
        assert group == "sde.plugins"
        return eps

    return _fake


def test_discover_plugins_loads_every_entry_point(monkeypatch) -> None:
    calls = []

    def register_a(registry: PluginRegistry) -> None:
        calls.append("a")
        registry.register_domain("domain_a", object)

    def register_b(registry: PluginRegistry) -> None:
        calls.append("b")
        registry.register_algorithm("algo_b", object)

    monkeypatch.setattr(
        "engine.discovery.entry_points",
        _fake_entry_points_returning(
            [_FakeEntryPoint("a", register_a), _FakeEntryPoint("b", register_b)]
        ),
    )

    registry = PluginRegistry()
    loaded = discover_plugins(registry)

    assert loaded == ["a", "b"]
    assert calls == ["a", "b"]
    assert registry.list_domains() == ["domain_a"]
    assert registry.list_algorithms() == ["algo_b"]


def test_discover_plugins_skips_broken_entry_point_with_warning(monkeypatch) -> None:
    def register_ok(registry: PluginRegistry) -> None:
        registry.register_domain("ok_domain", object)

    def register_broken(registry: PluginRegistry) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "engine.discovery.entry_points",
        _fake_entry_points_returning(
            [_FakeEntryPoint("broken", register_broken), _FakeEntryPoint("ok", register_ok)]
        ),
    )

    registry = PluginRegistry()
    with pytest.warns(UserWarning, match="broken"):
        loaded = discover_plugins(registry)

    assert loaded == ["ok"]
    assert registry.list_domains() == ["ok_domain"]


def test_discover_plugins_returns_empty_when_nothing_registered(monkeypatch) -> None:
    monkeypatch.setattr("engine.discovery.entry_points", _fake_entry_points_returning([]))

    registry = PluginRegistry()
    assert discover_plugins(registry) == []
    assert registry.list_domains() == []
    assert registry.list_algorithms() == []
