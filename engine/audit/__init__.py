"""Auditing whether a pipeline's wrapper contributes over its bare primitive.

Public surface for the subsystem specified in RFC-0001.
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
    paired_seed,
    run_arms,
)
from engine.audit.calibration import NullScaffold, OracleScaffold, WastefulScaffold
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
    "NullScaffold",
    "OracleScaffold",
    "Scaffold",
    "SearchOutcome",
    "Verdict",
    "WastefulScaffold",
    "assess_degeneracy",
    "audit",
    "equivalence_verdict",
    "paired_seed",
    "run_arms",
]
