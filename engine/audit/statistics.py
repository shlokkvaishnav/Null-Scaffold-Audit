"""Equivalence statistics for the scaffold-contribution audit (ENG-0001).

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


def equivalence_verdict(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    metric: str,
    margin: float,
    higher_is_better: bool = False,
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

    ci_low, ci_high = _interval(differences, confidence, resamples, seed)
    verdict = _resolve(ci_low, ci_high, margin)

    n = len(differences)
    spread = float(np.std(differences, ddof=1))
    # A (1 - 2*alpha) interval corresponds to a TOST at alpha per side.
    alpha = (1.0 - confidence) / 2.0

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
    )
