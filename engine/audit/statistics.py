"""Equivalence statistics for the scaffold-contribution audit.

The question this module answers is not "did the two arms differ?" but "were
they the same, within a margin agreed in advance?" Those require different
machinery, and using the first to answer the second is the defect the whole
audit exists to catch.

A difference test that returns ``p > 0.05`` licenses no conclusion at all. It
is compatible with a large real effect that the sample was too small to see.
Reporting it as evidence of no effect would let any underpowered comparison
certify a wrapper as contributing nothing -- confidently, plausibly, and
wrongly. So equivalence is established positively here, by requiring the whole
interval to sit inside the margin, and the power to detect that is reported on
every verdict rather than only when the answer is inconvenient.

This module knows nothing about what is being compared. It receives two
sequences of floats.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats

from engine.audit.verdict import MetricVerdict, Verdict

__all__ = ["equivalence_verdict"]


def _validate(
    treatment: Sequence[float],
    control: Sequence[float],
    margin: float,
    confidence: float,
) -> None:
    """Reject inputs whose verdict would be meaningless rather than merely wide."""
    if len(treatment) != len(control):
        raise ValueError(
            f"arms must be paired by seed, got {len(treatment)} treatment "
            f"and {len(control)} control observations"
        )
    if len(treatment) < 2:
        raise ValueError(
            f"at least 2 paired observations are required, got {len(treatment)}; "
            f"a bootstrap over one pair yields a meaningless interval, not a wide one"
        )
    if margin <= 0:
        raise ValueError(
            f"margin must be positive, got {margin}; a non-positive margin makes "
            f"equivalence unprovable and would silently yield permanent INCONCLUSIVE"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")

    for name, values in (("treatment", treatment), ("control", control)):
        for index, value in enumerate(values):
            if not math.isfinite(value):
                raise ValueError(f"{name}[{index}] is not finite: {value!r}")


def _tost_power(margin: float, spread: float, n: int, alpha: float) -> float:
    """Probability of establishing equivalence, assuming the true effect is zero.

    Reported on every verdict, not only on the inconclusive ones. A ``NULL``
    from a well-powered design and a ``NULL`` from a noisy three-seed run are
    different claims, and the enum member alone does not distinguish them.
    """
    if spread == 0.0:
        # Both arms are constant, so repeated sampling cannot disagree.
        return 1.0

    degrees_of_freedom = n - 1
    critical = stats.t.ppf(1.0 - alpha, degrees_of_freedom)
    noncentrality = margin * math.sqrt(n) / spread
    power = 2.0 * float(stats.t.cdf(noncentrality - critical, degrees_of_freedom)) - 1.0
    return float(min(max(power, 0.0), 1.0))


def _interval(
    differences: np.ndarray, confidence: float, resamples: int, seed: int
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval for the mean difference.

    Outcome metrics in this setting are skewed and heavy-tailed, so a
    parametric interval understates the tails in exactly the regime where the
    verdict is decided. BCa is degenerate when every difference is identical --
    there is nothing to resample -- so that case is handled here rather than
    left to surface as an exception from inside SciPy.
    """
    if float(np.ptp(differences)) == 0.0:
        point = float(differences[0])
        return point, point

    result = stats.bootstrap(
        (differences,),
        np.mean,
        confidence_level=confidence,
        n_resamples=resamples,
        method="BCa",
        random_state=np.random.default_rng(seed),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def _binary_counts(treatment: np.ndarray, control: np.ndarray) -> tuple[int, int, int]:
    """The paired 2x2 table, as ``(n, b, c)`` with ``b``/``c`` the discordant cells.

    ``b`` counts pairs the treatment got and the control did not; ``c`` counts
    the reverse. The concordant cells carry no information about the difference
    of proportions, so they are not returned.
    """
    values = np.concatenate([treatment, control])
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError(
            "a metric declared paired-binary must be 0 or 1 on every observation; "
            f"got values spanning [{values.min()}, {values.max()}]"
        )
    b = int(np.sum((treatment == 1.0) & (control == 0.0)))
    c = int(np.sum((treatment == 0.0) & (control == 1.0)))
    return len(treatment), b, c


def _tango_score(n: int, b: int, c: int, delta: float) -> float:
    """Tango's score statistic for the paired difference of proportions.

    The interval this generates is the one recommended for paired binomial
    proportions, and the reason it is used here rather than the bootstrap is a
    failure mode this audit cannot afford: the paired difference of a binary
    metric takes only the values -1, 0 and +1, so when no pair disagrees every
    resample is identical and the bootstrap interval collapses to a point. That
    reads as certainty. It is not -- twenty agreeing pairs are consistent with a
    true difference of about 0.12 -- and a point interval sits inside any margin,
    so it reports ``NULL``, the audit's one positive verdict, on no evidence.

    At ``delta = 0`` this reduces exactly to McNemar's statistic,
    ``(b - c) / sqrt(b + c)``, which is what the tests pin it against.
    """
    quadratic = 2 * n
    linear = -b - c + (2 * n - b + c) * delta
    constant = -c * delta * (1.0 - delta)

    discriminant = max(linear * linear - 4 * quadratic * constant, 0.0)
    # The constrained maximum-likelihood estimate of the "control succeeded,
    # treatment did not" cell probability, given a difference of exactly `delta`.
    constrained = (math.sqrt(discriminant) - linear) / (2 * quadratic)

    variance = n * (2.0 * constrained + delta * (1.0 - delta))
    if variance <= 0.0:
        # Reachable only at the boundary, where the constrained fit puts zero
        # mass on both discordant cells. The observed difference is then either
        # exactly `delta` or impossible under it.
        observed = b - c - n * delta
        if observed == 0.0:
            return 0.0
        return math.inf if observed > 0 else -math.inf
    return (b - c - n * delta) / math.sqrt(variance)


def _tango_interval(n: int, b: int, c: int, confidence: float) -> tuple[float, float]:
    """Invert the score statistic to a two-sided interval on the difference.

    ``_tango_score`` decreases monotonically in ``delta``, so each limit is
    found by bisection rather than by solving the quartic that eliminating the
    constrained estimate would produce. Fifty halvings take the bracket well
    below 1e-14, which is far finer than any margin this audit registers.
    """
    critical = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))

    def limit(target: float) -> float:
        low, high = -1.0, 1.0
        for _ in range(50):
            middle = (low + high) / 2.0
            if _tango_score(n, b, c, middle) > target:
                low = middle
            else:
                high = middle
        return (low + high) / 2.0

    return limit(critical), limit(-critical)


def _binary_spread(n: int, b: int, c: int) -> float:
    """Standard deviation of one paired difference, for the power calculation.

    Under a true difference of zero the variance of a single paired difference
    is the discordant rate, so the spread is its square root.

    When no pair disagrees, the observed rate is zero, and taking that at face
    value would report perfect power from a sample that observed nothing. The
    rate is unobserved rather than absent, so the rule of three supplies its 95%
    upper bound instead. That understates power rather than inventing it, which
    is the only direction of error this module is entitled to make.
    """
    discordant = b + c
    if discordant == 0:
        return math.sqrt(3.0 / n)
    return math.sqrt(discordant / n)


def _binary_claim_p_value(n: int, b: int, c: int, margin: float, verdict: Verdict) -> float | None:
    """``_claim_p_value``'s counterpart for a paired binary metric.

    Same three claims, tested with the same score statistic that produced the
    interval, so a verdict and its p-value cannot disagree about the evidence.
    """
    if verdict is Verdict.INCONCLUSIVE:
        return None

    at_upper = float(stats.norm.cdf(_tango_score(n, b, c, margin)))
    at_lower = float(stats.norm.cdf(_tango_score(n, b, c, -margin)))

    if verdict is Verdict.NULL:
        # Reject `difference <= -margin` and `difference >= +margin`; the claim
        # is only as strong as its weaker rejection.
        return max(1.0 - at_lower, at_upper)
    if verdict is Verdict.CONTRIBUTES:
        # Reject `difference <= +margin`, which large positive evidence does.
        return 1.0 - at_upper
    # HARMFUL: reject `difference >= -margin`.
    return at_lower


TARGET_POWER = 0.80
"""The power an equivalence claim is planned against, by convention."""


def required_sample_size(
    margin: float,
    spread: float,
    confidence: float = 0.90,
    target_power: float = TARGET_POWER,
    max_n: int = 100_000,
) -> int | None:
    """Smallest paired sample size whose TOST power reaches ``target_power``.

    Reported so ``INCONCLUSIVE`` stops being a surprise. An audit that keeps
    discovering after the fact that it could not tell is one that never asked
    how many seeds the question needed -- and the answer is knowable from the
    margin and the observed spread. "Inconclusive at 20 seeds, would need 96"
    is an instruction; "inconclusive" alone is only a disappointment.

    Uses the observed spread, so it is a retrospective estimate rather than a
    pre-registration: the spread is itself estimated from the sample it is
    describing, and at small n it is estimated badly. Treat the number as an
    order of magnitude for planning the next sweep, not as a guarantee.

    Returns None when even ``max_n`` observations would not reach the target,
    which means the margin is too tight for the noise and no amount of seeds
    will rescue it -- a finding about the design, not about the wrapper.
    """
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")
    if not 0.0 < target_power < 1.0:
        raise ValueError(f"target_power must lie in (0, 1), got {target_power}")
    if spread == 0.0:
        # Both arms are constant; repeated sampling cannot disagree, so the
        # smallest sample the statistics accept is already enough.
        return 2

    alpha = (1.0 - confidence) / 2.0
    if _tost_power(margin, spread, 2, alpha) >= target_power:
        return 2

    # Power is monotone in n, so find a bracket by doubling and then bisect.
    low, high = 2, 4
    while high <= max_n and _tost_power(margin, spread, high, alpha) < target_power:
        low, high = high, high * 2
    if high > max_n:
        return None

    while low < high:
        midpoint = (low + high) // 2
        if _tost_power(margin, spread, midpoint, alpha) >= target_power:
            high = midpoint
        else:
            low = midpoint + 1
    return low


def _claim_p_value(differences: np.ndarray, margin: float, verdict: Verdict) -> float | None:
    """Evidence for the claim this verdict makes, so it can be corrected.

    Each verdict asserts something different, so each needs its own test rather
    than one p-value pressed into three roles:

    - ``NULL`` claims equivalence, so the test is TOST: two one-sided tests
      against the margin, and the claim is only as strong as its weaker side.
    - ``CONTRIBUTES`` and ``HARMFUL`` claim a difference beyond the margin, so
      the test is one-sided in the direction claimed.
    - ``INCONCLUSIVE`` claims nothing, so there is nothing to correct. Returning
      None here rather than 1.0 keeps it out of the correction's family instead
      of padding that family with a claim nobody made.

    Parametric where the intervals are bootstrap, deliberately. The interval
    decides the verdict; this only ranks the claims against each other so Holm
    has an ordering. Inverting a BCa interval to get an exact p would cost a
    bootstrap per candidate level for no change in that ordering.
    """
    n = len(differences)
    spread = float(np.std(differences, ddof=1))
    mean = float(np.mean(differences))

    if verdict is Verdict.INCONCLUSIVE:
        return None

    if spread == 0.0:
        # Every observation agrees. The claim is certain or impossible; there is
        # no sampling variability left for a p-value to describe.
        if verdict is Verdict.NULL:
            return 0.0 if abs(mean) < margin else 1.0
        if verdict is Verdict.CONTRIBUTES:
            return 0.0 if mean > margin else 1.0
        return 0.0 if mean < -margin else 1.0

    standard_error = spread / math.sqrt(n)
    degrees_of_freedom = n - 1

    if verdict is Verdict.NULL:
        # H0 is non-equivalence on each side; the TOST p is the weaker rejection.
        p_above_lower = 1.0 - float(
            stats.t.cdf((mean + margin) / standard_error, degrees_of_freedom)
        )
        p_below_upper = float(stats.t.cdf((mean - margin) / standard_error, degrees_of_freedom))
        return float(max(p_above_lower, p_below_upper))

    if verdict is Verdict.CONTRIBUTES:
        return 1.0 - float(stats.t.cdf((mean - margin) / standard_error, degrees_of_freedom))

    return float(stats.t.cdf((mean + margin) / standard_error, degrees_of_freedom))


def _resolve(ci_low: float, ci_high: float, margin: float) -> Verdict:
    """Map an interval onto a verdict.

    Equivalence requires the entire interval to sit inside the margin. An
    interval that merely straddles zero is not equivalence -- it is an absence
    of evidence in either direction, which is ``INCONCLUSIVE``.
    """
    if ci_low > margin:
        return Verdict.CONTRIBUTES
    if ci_high < -margin:
        return Verdict.HARMFUL
    if -margin < ci_low and ci_high < margin:
        return Verdict.NULL
    return Verdict.INCONCLUSIVE


def _boundary_clearance_ratio(
    ci_low: float, ci_high: float, margin: float, verdict: Verdict
) -> float | None:
    """How many interval-widths past its margin a superiority verdict sits.

    ``None`` for anything but ``CONTRIBUTES``/``HARMFUL`` -- see
    ``MetricVerdict.boundary_clearance_ratio``'s docstring for why those are
    the only two verdicts this describes, and for the full formula and its
    provenance (issue #23).

    Uses whichever bound ``_resolve`` actually checked to reach the verdict,
    so this can never disagree with ``_resolve`` about which side mattered.
    A zero-width interval (every paired difference identical) has no ratio
    to report -- that case is also exactly what
    ``arms.py``'s ``_guard_vacuous_comparison`` withdraws to
    ``INCONCLUSIVE``, so a ``None`` here is never the only signal that
    something is off.
    """
    if verdict not in (Verdict.CONTRIBUTES, Verdict.HARMFUL):
        return None
    width = ci_high - ci_low
    if width <= 0.0:
        return None
    bound = ci_low if verdict is Verdict.CONTRIBUTES else -ci_high
    return (bound - margin) / width


def equivalence_verdict(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    metric: str,
    margin: float,
    higher_is_better: bool = False,
    paired_binary: bool = False,
    confidence: float = 0.90,
    resamples: int = 10_000,
    seed: int = 0,
) -> MetricVerdict:
    """Decide whether the treatment arm differs from the control within ``margin``.

    Observations are paired by seed: ``treatment[i]`` and ``control[i]`` are the
    same problem under the same seed, differing only in whether the wrapper was
    present. Pairing removes between-problem variance, which is usually far
    larger than the effect being measured.

    ``margin`` is the largest difference considered practically uninteresting,
    and it must be chosen before the data is seen. Choosing it afterwards turns
    the audit into a search for a threshold that produces the preferred answer.

    ``higher_is_better`` orients the metric. Getting it wrong inverts
    ``CONTRIBUTES`` and ``HARMFUL`` while leaving every number plausible, so it
    is an explicit argument rather than something inferred from the metric name.

    ``paired_binary`` selects the machinery for a metric whose observations are
    successes and failures rather than measurements -- a recovery rate, say. It
    is *declared* by the caller and never inferred from the values, because a
    continuous metric that happened to come back all-zeros on one sweep is still
    continuous, and a rule that sniffed the data would silently switch tests
    between sweeps of the same design. See ``_tango_score`` for why the default
    machinery is not merely imprecise on such a metric but actively wrong.

    The function is pure and seeded: the same inputs return the same verdict,
    because a verdict that moved between runs could not be published next to
    the number it qualifies.

    Raises:
        ValueError: on unpaired arms, fewer than two pairs, a non-positive
            margin, a confidence outside (0, 1), or any non-finite observation.
    """
    _validate(treatment, control, margin, confidence)

    treatment_values = np.asarray(treatment, dtype=float)
    control_values = np.asarray(control, dtype=float)

    # Oriented so that positive always means the treatment arm did better,
    # whichever way the raw metric runs.
    differences = treatment_values - control_values
    if not higher_is_better:
        differences = -differences

    n = len(differences)
    # A (1 - 2*alpha) interval corresponds to a TOST at alpha per side.
    alpha = (1.0 - confidence) / 2.0

    if paired_binary:
        # Orientation is already applied above, so the arms are re-derived from
        # the oriented difference rather than from the raw inputs: b counts the
        # pairs where the treatment did better by the metric's own direction.
        b = int(np.sum(differences == 1.0))
        c = int(np.sum(differences == -1.0))
        _binary_counts(treatment_values, control_values)  # validates 0/1 inputs
        ci_low, ci_high = _tango_interval(n, b, c, confidence)
        verdict = _resolve(ci_low, ci_high, margin)
        spread = _binary_spread(n, b, c)
        p_value = _binary_claim_p_value(n, b, c, margin, verdict)
        test = "Tango score interval on the paired difference of proportions"
    else:
        ci_low, ci_high = _interval(differences, confidence, resamples, seed)
        verdict = _resolve(ci_low, ci_high, margin)
        spread = float(np.std(differences, ddof=1))
        p_value = _claim_p_value(differences, margin, verdict)
        test = "BCa bootstrap interval on the paired mean difference"

    return MetricVerdict(
        metric=metric,
        verdict=verdict,
        observed_difference=float(np.mean(differences)),
        ci_low=ci_low,
        ci_high=ci_high,
        margin=margin,
        power=_tost_power(margin, spread, n, alpha),
        n=n,
        higher_is_better=higher_is_better,
        p_value=p_value,
        n_for_target_power=required_sample_size(margin, spread, confidence),
        boundary_clearance_ratio=_boundary_clearance_ratio(ci_low, ci_high, margin, verdict),
        test=test,
    )
