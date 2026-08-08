"""Auditing whether a pipeline's wrapper contributes over its bare primitive.

Public surface for the subsystem specified in RFC-0001 and decided in ADR-0001.
Names exported here are a compatibility commitment; anything absent from
``__all__`` is an internal detail and may change without notice.
"""

from __future__ import annotations

from engine.audit.statistics import equivalence_verdict
from engine.audit.verdict import MetricVerdict, Verdict

__all__ = ["MetricVerdict", "Verdict", "equivalence_verdict"]
