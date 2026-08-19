# Archive: physics equation discovery

This is the project's original research direction: an agentic symbolic-regression
system (`plugins/physics/`) rediscovering closed-form physics equations from the
AI-Feynman benchmark, plus a second domain (`plugins/synthetic/`) used to validate
the plugin interfaces. It is archived, not deleted, and not maintained going
forward — see the project root README for why.

**This subtree is kept for provenance, not for reuse.** In particular:

- `plugins/physics/scaffold/generator.py` produced the character-identical output
  across all three "refine" iterations that motivated `engine/audit/` in the first
  place — the bug is described in `docs/rfc/RFC-0001-null-scaffold-audit.md` at the
  project root, and the fix is visible in this file's own comments
  (`_search_seed`).
- `results/` under this directory holds every physics/synthetic-domain audit,
  calibration, ceiling, and feasibility run produced before the pivot, including
  the runs that first showed the scaffold was statistically indistinguishable from
  — or worse than — plain restarts.

**Not guaranteed importable from the active tree.** `pyproject.toml`'s
`sde.plugins` entry points for `feynman_physics` and `synthetic_regression` were
removed when this moved here. Re-running anything in this subtree requires
reinstalling those entry points and pointing them at this path — this archive
exists to be read, not necessarily to be re-run.

**What stayed active, and why:** `engine/audit/` (the equivalence-testing,
budget-matched-restart, oracle-ceiling, and feasibility-gate machinery) is
domain-agnostic and was never physics-specific — it moved nowhere, because it's
the part of the project the new research direction (auditing Neural Architecture
Search controllers against budget-matched random search) reuses directly.
