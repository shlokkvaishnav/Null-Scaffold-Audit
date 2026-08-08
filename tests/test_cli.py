"""Tests for the `sde` CLI (cli.main): list-plugins and run against the Feynman plugin.

_default_registry() now discovers plugins via the "sde.plugins" entry-point
group (engine.discovery.discover_plugins) instead of importing
physics_discovery by name -- see docs/PLUGIN_GUIDE.md. Entry points are
packaging metadata: they only exist once this repo is actually installed
(`pip install -e .`, as CI and the Docker build both do), not from a bare
source checkout. Skip this whole module when that isn't the case here,
rather than asserting on an empty registry -- verify via Docker/CI instead.
"""

from __future__ import annotations

import json
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from typer.testing import CliRunner

from engine.discovery import ENTRY_POINT_GROUP

if not list(entry_points(group=ENTRY_POINT_GROUP)):
    pytest.skip(
        "No 'sde.plugins' entry points found -- this repo isn't pip-installed in this "
        "environment (entry points require `pip install -e .`, as CI/Docker do). "
        "Verify via Docker/CI instead.",
        allow_module_level=True,
    )

from cli.main import app

try:
    # cli.main discovers plugins lazily inside _default_registry(), so probe
    # the same downstream import here rather than at `from cli.main import app`.
    from physics_discovery.plugins import feynman  # noqa: F401
except ImportError as exc:
    pytest.skip(
        f"physics_discovery.plugins.feynman not importable here: {exc}", allow_module_level=True
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_list_plugins_reports_feynman_plugin() -> None:
    result = runner.invoke(app, ["list-plugins"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["domains"] == ["feynman_physics", "synthetic_regression"]
    assert "symbolic_regression" in payload["algorithms"]
    assert "gbm_baseline" in payload["algorithms"]


def test_run_command_executes_orchestrator(tmp_path: Path) -> None:
    config_path = tmp_path / "gbm_run.yaml"
    config_path.write_text(
        """
domain: feynman_physics
algorithm: gbm_baseline
domain_kwargs:
  equation_id: coulomb_force
  n_samples: 60
  seed: 0
algorithm_kwargs:
  random_state: 0
"""
    )

    result = runner.invoke(app, ["run", str(config_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["domain"] == "feynman_physics"
    assert payload["algorithm"] == "gbm_baseline"
    assert payload["equation"] is None
    assert "rmse" in payload["metrics"]
