"""Margin-sensitivity sweep on the already-run nas_search self-audit, per SPEC.md.

    python analysis/nas_margin_sensitivity/run_sweep.py

Loads the 30 paired `valid_accuracy` observations already sitting in
`results/nas_search_self_audit/audit.json` (issue #11 / PR #12) and re-tests
them against a swept range of equivalence margins, using
`engine.audit.statistics`'s own machinery. No NATS-Bench evaluation is
re-run and no new data is collected -- this is exactly what makes the branch
`analysis/`, not `experiment/`.

Reuses two of that module's private functions directly (`_interval`,
`_resolve`) rather than re-implementing them: `_interval` produces the BCa
bootstrap confidence interval on the paired mean difference, and `_resolve`
maps an interval onto a verdict given a margin. Calling `_interval` once and
sweeping `_resolve` across margins -- rather than calling
`equivalence_verdict()` once per swept margin -- is deliberate and is what
SPEC.md's "Confounds considered" section requires: the interval does not
depend on the margin, only which side of +/-margin it lands on does, so
recomputing it per margin would mean thirty-thousand-plus independent
bootstrap resamples whose resampling noise could produce a spurious verdict
flip near a crossover that has nothing to do with the margin itself.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from engine.audit.statistics import _interval, _resolve
from engine.audit.verdict import Verdict

AUDIT_SOURCE = REPO_ROOT / "results" / "nas_search_self_audit" / "audit.json"
OUT = REPO_ROOT / "results" / "nas_margin_sensitivity"

METRIC = "valid_accuracy"
HIGHER_IS_BETTER = True
CONFIDENCE = 0.90
RESAMPLES = 10_000
SEED = 0
PRE_REGISTERED_MARGIN = 0.3

# Coarse sweep: cheap since it touches no new data, per SPEC.md. Range chosen
# to bracket the crossover regions implied by PR #12's already-reported CI
# ([-0.197, -0.008]pp) -- not chosen after seeing where a crossover happens
# to look interesting (SPEC.md, "Confounds considered").
SWEEP_MIN = 0.001
SWEEP_MAX = 1.2
SWEEP_STEP = 0.001

BISECTION_TOLERANCE = 1e-6
"""Refines a coarse-sweep crossover bracket to this precision, well past the
'>= 2 significant figures' SPEC.md asks for."""


def _display_path(path: Path) -> str:
    """`path` relative to the repo root when possible, else its raw form.

    Falls back rather than raising: tests point `AUDIT_SOURCE` at a tmp
    directory outside `REPO_ROOT`, where `relative_to` has no answer.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_paired_metrics() -> tuple[list[float], list[float]]:
    payload = json.loads(AUDIT_SOURCE.read_text(encoding="utf-8"))
    arms = payload["arms"]
    treatment = arms["treatment_metrics"]
    control = arms["control_metrics"]
    if len(treatment) != len(control):
        raise ValueError(
            f"expected paired arms, got {len(treatment)} treatment vs {len(control)} control"
        )
    return treatment, control


def _oriented_differences(treatment: list[float], control: list[float]) -> np.ndarray:
    differences = np.asarray(treatment, dtype=float) - np.asarray(control, dtype=float)
    if not HIGHER_IS_BETTER:
        differences = -differences
    return differences


def _refine_crossover(
    low_margin: float, high_margin: float, ci_low: float, ci_high: float
) -> float:
    """Bisect a coarse-sweep bracket to `BISECTION_TOLERANCE`.

    `_resolve`'s verdict is monotone non-decreasing in margin along the
    ordering HARMFUL < INCONCLUSIVE < NULL < CONTRIBUTES for a fixed interval
    (a wider +/-margin can only make the interval fit more easily inside it),
    so a bracket that changes verdict brackets exactly one crossover.
    """
    order = {Verdict.HARMFUL: 0, Verdict.INCONCLUSIVE: 1, Verdict.NULL: 2, Verdict.CONTRIBUTES: 3}
    low_rank = order[_resolve(ci_low, ci_high, low_margin)]
    while high_margin - low_margin > BISECTION_TOLERANCE:
        mid = (low_margin + high_margin) / 2.0
        if order[_resolve(ci_low, ci_high, mid)] > low_rank:
            high_margin = mid
        else:
            low_margin = mid
    return high_margin


def main() -> int:
    treatment, control = _load_paired_metrics()
    differences = _oriented_differences(treatment, control)

    ci_low, ci_high = _interval(differences, CONFIDENCE, RESAMPLES, SEED)
    print(f"[margin-sensitivity] recomputed CI: [{ci_low:.6f}, {ci_high:.6f}]")

    # Sanity check: this must reproduce PR #12's reported interval exactly,
    # since it is the same computation (same paired data, same confidence,
    # same resample count, same seed) -- if it doesn't, the sweep below would
    # be silently answering a different question than the one asked.
    original = json.loads(AUDIT_SOURCE.read_text(encoding="utf-8"))
    original_verdict = original["per_metric"][METRIC]
    assert abs(ci_low - original_verdict["ci_low"]) < 1e-9, (
        "recomputed ci_low does not match PR #12"
    )
    assert abs(ci_high - original_verdict["ci_high"]) < 1e-9, (
        "recomputed ci_high does not match PR #12"
    )
    print("[margin-sensitivity] CI matches results/nas_search_self_audit/audit.json exactly")

    margins = np.arange(SWEEP_MIN, SWEEP_MAX + SWEEP_STEP, SWEEP_STEP)
    rows: list[dict[str, float | str]] = [
        {"margin": round(float(m), 6), "verdict": _resolve(ci_low, ci_high, float(m)).value}
        for m in margins
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["margin", "verdict"])
        writer.writeheader()
        writer.writerows(rows)

    # Crossovers: scan the coarse sweep for verdict changes, then bisect each
    # bracket to BISECTION_TOLERANCE.
    crossovers: list[dict[str, float | str]] = []
    for i in range(1, len(rows)):
        if rows[i]["verdict"] != rows[i - 1]["verdict"]:
            refined = _refine_crossover(
                float(rows[i - 1]["margin"]), float(rows[i]["margin"]), ci_low, ci_high
            )
            crossovers.append(
                {
                    "from_verdict": rows[i - 1]["verdict"],
                    "to_verdict": rows[i]["verdict"],
                    "margin": refined,
                }
            )

    pre_registered_verdict = _resolve(ci_low, ci_high, PRE_REGISTERED_MARGIN).value
    nearest = (
        min(crossovers, key=lambda c: abs(float(c["margin"]) - PRE_REGISTERED_MARGIN))
        if crossovers
        else None
    )

    report = {
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": CONFIDENCE,
        "resamples": RESAMPLES,
        "seed": SEED,
        "n": len(treatment),
        "sweep_range": [SWEEP_MIN, SWEEP_MAX, SWEEP_STEP],
        "pre_registered_margin": PRE_REGISTERED_MARGIN,
        "pre_registered_verdict": pre_registered_verdict,
        "crossovers": crossovers,
        "nearest_crossover_to_pre_registered_margin": nearest,
        "distance_to_nearest_crossover": (
            abs(float(nearest["margin"]) - PRE_REGISTERED_MARGIN) if nearest else None
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    source_line = (
        f"Source: `{_display_path(AUDIT_SOURCE)}` (issue #11 / PR #12), "
        f"n={len(treatment)} paired `{METRIC}` observations, held fixed."
    )
    ci_line = (
        f"90% BCa bootstrap CI on the paired difference (higher_is_better="
        f"{HIGHER_IS_BETTER}, resamples={RESAMPLES}, seed={SEED}): "
        f"[{ci_low:.4f}, {ci_high:.4f}] percentage points."
    )
    margin_line = (
        f"Pre-registered margin (issue #11): +/-{PRE_REGISTERED_MARGIN}pp -> "
        f"verdict **{pre_registered_verdict}** (matches PR #12's reported verdict)."
    )
    lines = [
        "# NAS RandomSearch self-audit: margin-sensitivity sweep",
        "",
        source_line,
        "",
        ci_line,
        "",
        margin_line,
        "",
        "## Crossovers",
        "",
    ]
    if crossovers:
        assert nearest is not None  # crossovers non-empty => nearest was computed above
        for c in crossovers:
            lines.append(
                f"- {c['from_verdict']} -> {c['to_verdict']} at margin = {float(c['margin']):.4f}pp"
            )
        distance = float(report["distance_to_nearest_crossover"])  # type: ignore[arg-type]
        nearest_margin = float(nearest["margin"])
        lines.append("")
        lines.append(
            f"Nearest crossover to the pre-registered margin: {nearest_margin:.4f}pp "
            f"({nearest['from_verdict']} <-> {nearest['to_verdict']}), a distance of "
            f"{distance:.4f}pp ({PRE_REGISTERED_MARGIN / nearest_margin:.2f}x)."
        )
    else:
        lines.append(
            f"None found in [{SWEEP_MIN}, {SWEEP_MAX}]pp -- verdict is robust across this range."
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"[margin-sensitivity] pre-registered margin {PRE_REGISTERED_MARGIN}pp -> {pre_registered_verdict}"
    )
    for c in crossovers:
        print(
            f"[margin-sensitivity] crossover: {c['from_verdict']} -> {c['to_verdict']} "
            f"at {float(c['margin']):.4f}pp"
        )
    print(
        f"[margin-sensitivity] wrote {OUT / 'summary.csv'}, {OUT / 'summary.md'}, {OUT / 'report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
