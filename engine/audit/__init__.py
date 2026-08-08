"""Auditing whether a pipeline's wrapper contributes over its bare primitive.

Public surface for the subsystem specified in RFC-0001 and decided in ADR-0001.
Names exported here are a compatibility commitment; anything absent from
``__all__`` is an internal detail and may change without notice.
"""

from __future__ import annotations

from engine.audit.arms import (
    ArmOutcomes,
    AuditReport,
    BaseSearcher,
    Budget,
    NotSeparableError,
    Scaffold,
    SearchOutcome,
    audit,
    run_arms,
)
from engine.audit.statistics import equivalence_verdict
from engine.audit.verdict import MetricVerdict, Verdict

__all__ = [
    "ArmOutcomes",
    "AuditReport",
    "BaseSearcher",
    "Budget",
    "MetricVerdict",
    "NotSeparableError",
    "Scaffold",
    "SearchOutcome",
    "Verdict",
    "audit",
    "equivalence_verdict",
    "run_arms",
]
