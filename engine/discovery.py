"""Dynamic plugin discovery via Python packaging entry points.

This is what makes plugins genuinely pluggable: nothing in `engine/` or
`cli/` imports a concrete plugin module (like `physics_discovery.plugins.
feynman`) by name. Instead, any *installed* Python package can register SDE
plugins by declaring an entry point under the `"sde.plugins"` group in its
own `pyproject.toml`:

    [project.entry-points."sde.plugins"]
    my_domain = "my_package.plugins:register"

where `my_package.plugins.register` has the same signature every plugin
module in this repo already uses:

    def register(registry: PluginRegistry) -> None: ...

`discover_plugins` finds every such entry point across every installed
package (this repo's own plugins included -- see the `[project.entry-
points."sde.plugins"]` table in this repo's pyproject.toml) and calls each
one's `register(registry)`. Add a plugin by installing a package with the
right entry point; no change to this repo required.

Entry points only exist once a package is actually installed (`pip install`,
including editable installs) -- they are packaging metadata, not something
importable from a bare source checkout. See docs/DEVELOPMENT.md.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points

from engine.registry import PluginRegistry

ENTRY_POINT_GROUP = "sde.plugins"


def discover_plugins(registry: PluginRegistry, group: str = ENTRY_POINT_GROUP) -> list[str]:
    """Load and call register(registry) for every installed entry point in `group`.

    Returns the names of the entry points that loaded successfully, in
    discovery order. A single broken entry point (import error, or a
    register() call that raises) is skipped with a warning rather than
    aborting discovery of every other installed plugin.
    """
    loaded: list[str] = []
    for ep in entry_points(group=group):
        try:
            register_fn = ep.load()
            register_fn(registry)
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not break discovery
            warnings.warn(f"Failed to load SDE plugin entry point {ep.name!r}: {exc}", stacklevel=2)
            continue
        loaded.append(ep.name)
    return loaded
