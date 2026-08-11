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
from engine.audit.degeneracy import DegeneracyReport, assess_degeneracy
from engine.audit.problem import AuditProblem, AuditProblemSource
from engine.audit.statistics import equivalence_verdict
from engine.audit.verdict import MetricVerdict, Verdict

__all__ = [
    "ArmOutcomes",
    "AuditProblem",
    "AuditProblemSource",
    "AuditReport",
    "BaseSearcher",
    "Budget",
    "DegeneracyReport",
    "MetricVerdict",
    "NotSeparableError",
    "Scaffold",
    "SearchOutcome",
    "Verdict",
    "assess_degeneracy",
    "audit",
    "equivalence_verdict",
    "run_arms",
]
