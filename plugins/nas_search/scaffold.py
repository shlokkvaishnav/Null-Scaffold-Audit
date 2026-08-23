"""IdentityRestartScaffold: this plugin's null scaffold, per SPEC.md.

``engine.audit.calibration.NullScaffold`` already implements "the same
search, the same budget, the same rule" for any ``BaseSearcher`` -- it draws
``restarts`` independent searches and selects among them with the base
searcher's own ``select()``, which is exactly what the control arm does, so
the true difference between the two arms is zero on every metric by
construction (see its docstring for why this is a harder and more useful
test than sharing seeds with the control outright).

This subclass exists only to give it this plugin's own name, matching
SPEC.md's ``IdentityRestartScaffold`` and the issue's explicit instruction
that this be "the domain-specific analogue of calibration.py's
NullScaffold, not a new scaffold design" -- so the audit sweep reports a
scaffold that belongs to ``plugins/nas_search``, not a calibration fixture
borrowed as if it were a proposed pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.audit.calibration import NullScaffold


@dataclass
class IdentityRestartScaffold(NullScaffold):
    name: str = "nas_search.identity_restart"
