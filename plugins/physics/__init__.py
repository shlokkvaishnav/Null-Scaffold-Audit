"""The physics domain plugin, and the pipeline the audit was built to measure.

Layout, and why:

- `plugin.py`          the `DomainPlugin`/`AlgorithmPlugin` adapters, entry point
- `feynman_loader.py`  the AI-Feynman equation set
- `scaffold/`          the DiscoveryAgent loop
- `audit_adapter.py`   exposes that loop to `engine.audit`
- `inference/`         the variational EM apparatus (needs torch)
- `api/`               a single-process demo service over the loop

`scaffold/` sits here rather than in the core for a measured reason. The
null-scaffold audit compared the loop against its own unwrapped search
primitive at matched budgets and found it HARMFUL or INCONCLUSIVE on 8 of 8
problems, degenerate on every problem at every budget, and CONTRIBUTES on none.
Promoting it into `engine/` would make a refuted design the foundation, so it
stays plugin-side as a documented negative result. Re-run the audit before
treating that as settled -- and if the numbers change, this docstring is the
defect.

Nothing is re-exported here: the entry point names `plugins.physics.plugin`
directly, and importing this package should not drag in gplearn and sklearn.
"""
