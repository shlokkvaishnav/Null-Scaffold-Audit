"""`sde` CLI: a single entry point over the engine orchestrator and registered plugins.

Usage:
    sde list-plugins
    sde run configs/run/coulomb_symbolic.yaml

A `benchmark` command used to live here, wrapping a paper-reproduction script
over `configs/paper/*.yaml`. It was removed along with that script: the figures
it produced belonged to a framing this project retracted, and one of its configs
keyed a figure on the belief-entropy of the agent loop the audit refuted. The
audit runner under `scripts/` is the entry point that survived, and it stays a
script rather than a subcommand -- it takes a pre-registered margin set and
writes artifacts, which suits a recorded command line better than a one-liner.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from engine.config import load_run_config
from engine.discovery import discover_plugins
from engine.orchestrator import DiscoveryOrchestrator
from engine.registry import PluginRegistry

app = typer.Typer(add_completion=False, help="Scientific Discovery Engine CLI.")


def _default_registry() -> PluginRegistry:
    """Build a registry from every installed SDE plugin entry point.

    No plugin module is imported by name here. Anybody can add their own
    domain or algorithm without touching this file: install a package that
    declares an entry point in the "sde.plugins" group (see
    docs/PLUGIN_GUIDE.md) and it is picked up automatically. This repo's own
    plugins (plugins.physics.plugin / .synthetic) are registered
    the same way, via entry points in this repo's own pyproject.toml -- not
    special-cased here.
    """
    registry = PluginRegistry()
    discover_plugins(registry)
    return registry


@app.command("list-plugins")
def list_plugins() -> None:
    """List every domain and algorithm plugin available to the CLI by default."""
    registry = _default_registry()
    typer.echo(
        json.dumps(
            {"domains": registry.list_domains(), "algorithms": registry.list_algorithms()},
            indent=2,
        )
    )


# Typer reads its argument metadata from the default value, which means calling
# typer.Argument() in the signature -- the pattern flake8-bugbear rejects,
# because a call evaluated once at import time is a trap when the value is
# mutable. These are immutable metadata objects, so the hazard does not apply;
# hoisting them to module scope satisfies the rule without disguising anything.
_RUN_CONFIG_ARGUMENT = typer.Argument(..., help="Path to a RunConfig YAML file.")


@app.command("run")
def run(config_path: Path = _RUN_CONFIG_ARGUMENT) -> None:
    """Run one domain+algorithm plugin pair through the orchestrator."""
    config = load_run_config(config_path)
    registry = _default_registry()
    orchestrator = DiscoveryOrchestrator(registry)
    result = orchestrator.run(config.to_experiment_config())
    typer.echo(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    app()
