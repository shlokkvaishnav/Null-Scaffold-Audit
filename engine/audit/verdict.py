"""Verdict records for the scaffold-contribution audit.

The enumeration is closed. Verdicts are published beside the numbers they
qualify and are compared across methods and across time, so a new member is a
change to a published data schema rather than a convenience -- it is added by
a recorded decision, not by editing this file.

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
    """Achieved power against ``margin``, in [0, 1], at the observed variance.

    This answers one specific question: the probability of establishing
    *equivalence* at this margin and spread, assuming the true effect is
    zero (``_tost_power``'s own docstring). That is the right question for a
    ``NULL`` or ``INCONCLUSIVE`` verdict, both of which are claims (or
    non-claims) about equivalence. It is not the right question for
    ``CONTRIBUTES``/``HARMFUL``, which are one-sided superiority/
    inferiority claims -- a well-powered ``CONTRIBUTES`` row can report a
    low ``power`` here (see issue #23) simply because the true effect is
    not zero, which is exactly what a superiority claim asserts. Read
    ``power``/``n_for_target_power`` on those two verdicts as "what this
    row would need to also certify equivalence" -- an unrelated question --
    not as evidence about the superiority claim actually made. Use
    ``boundary_clearance_ratio`` for that.
    """

    n: int
    higher_is_better: bool

    p_value: float | None = None
    """Evidence for the claim this verdict makes, before correction.

    For ``NULL`` this is the TOST p-value -- the larger of the two one-sided
    tests against the margin. For ``CONTRIBUTES`` and ``HARMFUL`` it is the
    one-sided test against the margin in the claimed direction.
    ``INCONCLUSIVE`` makes no claim, so it has none.
    """

    adjusted_p_value: float | None = None
    """``p_value`` after correcting across the metrics audited together.

    AUDIT_METHODOLOGY.md §4.2 requires this. Three metrics tested at 0.05 each is not
    a 0.05 procedure, and the audit exists to refuse exactly that kind of
    quietly-inflated confidence.
    """

    n_for_target_power: int | None = None
    """Paired observations needed to reach the conventional 80% power, at the
    observed spread. ``None`` means no attainable sample size would, which is a
    finding about the margin rather than about the wrapper.

    Present so ``INCONCLUSIVE`` carries an instruction rather than only a
    disappointment: "could not tell at 20 seeds, would need 96" is actionable.
    """

    boundary_clearance_ratio: float | None = None
    """How decisively a ``CONTRIBUTES``/``HARMFUL`` verdict cleared its
    margin, in units of the interval's own width. ``None`` for every other
    verdict -- ``NULL``/``INCONCLUSIVE`` are equivalence questions, already
    served by ``power``/``n_for_target_power`` (see that field's docstring).

    Defined as ``(clearance - margin) / (ci_high - ci_low)``, where
    ``clearance`` is whichever CI bound actually decided the verdict
    (``ci_low`` for ``CONTRIBUTES``, ``-ci_high`` for ``HARMFUL`` --
    matching ``_resolve``'s own boundary checks). A large value means the
    bound sits many interval-widths past the margin -- decisive at this
    ``n``; a value near zero means the bound only just cleared it, and a
    re-run at higher ``n`` (which narrows the interval) is the number this
    project's own practice used to reach for by hand before this field
    existed (issue #23; see ``STEPSIZE_SPEC.md`` and issues #19/#21 for the
    manual version of this reasoning). Undefined (``None``) on a
    zero-width interval (every paired difference identical), where the
    ratio's denominator vanishes and the vacuous-comparison guard
    (``arms.py``'s ``_guard_vacuous_comparison``) already withdraws the
    verdict to ``INCONCLUSIVE`` regardless.

    Normalized by the interval's full width, not its half-width, despite
    issue #23's prose describing "CI-half-widths" -- the issue's own
    concrete formula divides by ``ci_high - ci_low`` (the full width), and
    this implements that formula literally rather than the looser prose
    description; the discrepancy is noted here rather than silently
    resolved one way. A reader wanting the half-width version multiplies
    this value by 2.
    """

    test: str | None = None
    """Which procedure produced ``ci_low``, ``ci_high`` and ``p_value``.

    Recorded because the audit does not use one test for every metric: a
    measurement and a success rate need different machinery, and a reader
    comparing two verdicts is entitled to know whether they were reached the
    same way. It travels with the number rather than living in prose.
    """

    correction: str | None = None
    """Which correction produced ``adjusted_p_value``, named in the record.

    The RFC's wording is that the correction is "stated in the report rather
    than left implicit", so it travels with the number rather than living in
    prose someone has to go and find.
    """
