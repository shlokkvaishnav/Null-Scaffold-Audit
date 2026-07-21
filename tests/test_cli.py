"""Tests for the `sde` CLI (cli.main): list-plugins and run against the Feynman plugin.

Requires sklearn (used transitively by the Feynman plugin the CLI's default
registry loads) to be importable; skipped otherwise rather than erroring
collection -- see tests/test_feynman_plugin.py for the same pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli.main import app

try:
    # cli.main imports the Feynman plugin lazily inside _default_registry(),
    # so probe that same import here rather than at `from cli.main import app`.
    from physics_discovery.plugins import feynman  # noqa: F401
except ImportError as exc:
    pytest.skip(f"physics_discovery.plugins.feynman not importable here: {exc}", allow_module_level=True)

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
