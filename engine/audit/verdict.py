"""Verdict records for the scaffold-contribution audit (ENG-0001).

The enumeration is closed. Verdicts are published beside the numbers they
qualify and are compared across methods and across time, so a new member is a
change to a published data schema rather than a convenience -- it is added by
amending ADR-0001, not by editing this file.

The distinction that carries the weight is between ``NULL`` and
``INCONCLUSIVE``. ``NULL`` asserts that the wrapper contributed nothing within
a stated margin. ``INCONCLUSIVE`` asserts that the comparison could not tell.
Collapsing the second into the first is the error this whole subsystem exists
to prevent, which is why they are separate members rather than one member and
a confidence score that a reader may or may not look at.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    """The outcome of comparing a wrapped pipeline against its bare primitive."""

    CONTRIBUTES = "CONTRIBUTES"
    """The wrapper measurably outperformed the primitive beyond the margin."""

    NULL = "NULL"
    """The two arms were equivalent within the margin. A positive finding."""

    HARMFUL = "HARMFUL"
    """The wrapper measurably underperformed the primitive beyond the margin."""

    INCONCLUSIVE = "INCONCLUSIVE"
    """The comparison could not distinguish the above. Never reported as NULL."""

    NOT_SEPARABLE = "NOT_SEPARABLE"
    """The pipeline could not expose a bare primitive, so no control arm exists.

    Assigned by the caller, never by the statistics: it is a property of a
    pipeline's structure, not of a sample.
    """


@dataclass(frozen=True)
class MetricVerdict:
    """One metric's verdict, with the evidence that justifies it.

    Every field here is part of the claim. A verdict without its interval,
    margin, and power is an assertion rather than a result -- a ``NULL`` from
    three noisy runs and a ``NULL`` from fifty tight ones are different claims,
    and a reader who sees only the enum member cannot tell them apart.
    """

    metric: str
    verdict: Verdict

    observed_difference: float
    """Oriented improvement of treatment over control: positive always means the
    treatment arm did better, whichever direction the raw metric runs."""

    ci_low: float
    ci_high: float
    """Bootstrap interval bounds on ``observed_difference``, at ``confidence``."""

    margin: float
    """The pre-registered equivalence margin. Chosen before the data was seen."""

    power: float
    """Achieved power against ``margin``, in [0, 1], at the observed variance."""

    n: int
    higher_is_better: bool
