"""Domain plugins: everything the engine deliberately does not know.

Each subpackage supplies one scientific domain -- its data, its constraints,
and whatever pipeline it wants audited -- behind the protocols in
`engine.plugin`. The engine may not import from here, and that rule is checked
mechanically: `tools/check_domain_independence.py` lists `plugins` among its
forbidden roots.

Nothing is re-exported at this level, on purpose. A plugin is discovered
through its entry point in `pyproject.toml`, not by importing this package, so
adding a domain requires no edit here and none in `engine/`.
"""
