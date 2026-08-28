"""Tests for withdrawing a margin-degenerate verdict to INCONCLUSIVE (issue
#29, ``engine/audit/GATE_MARGIN_DEGENERACY_SPEC.md``).

Issue #27 added the *detection* (`MetricVerdict.margin_degeneracy`,
diagnostic only at the time). This issue decided the *gating* question left
open there. The unit tests pin `_guard_margin_degeneracy` directly
(mirroring `test_audit_boundary_clearance.py`'s precedent for testing a
private arms.py helper against constructed `MetricVerdict`s, so the inputs
are exact rather than left to sampling luck). The retrospective test is the
actual validation this branch's SPEC promised: re-running gating against
every row `tests/test_audit_margin_degeneracy.py` already covers must
change nothing for any already-trusted row and must withdraw both
already-untrustworthy Griewank readings to INCONCLUSIVE.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from engine.audit import SearchOutcome, Verdict, audit
from engine.audit.arms import _guard_margin_degeneracy
from engine.audit.margin_degeneracy import assess_margin_degeneracy
from engine.audit.statistics import equivalence_verdict

REPO_ROOT = Path(__file__).resolve().parents[1]

# Same artifact set and trust labels as tests/test_audit_margin_degeneracy.py
# -- duplicated rather than imported, so this file's own claims stand on
# their own re-derivation of "which rows are which," not on trusting the
# other test module got it right.
_MULTI_FUNCTION_ARTIFACTS = [
    "results/basinhopping_audit/audit.json",
    "results/basinhopping_audit_stepsize_scaling/audit.json",
    "results/basinhopping_audit_seed_cap_fix/run_audit/audit.json",
    "results/basinhopping_audit_seed_cap_fix/run_stepsize_experiment/audit.json",
]
_SINGLE_FUNCTION_ARTIFACTS = [
    "results/basinhopping_audit_ackley_power/audit.json",
    "results/basinhopping_audit_rastrigin_power/audit.json",
]
_UNTRUSTED_GRIEWANK_FILES = {
    "results/basinhopping_audit/audit.json",
    "results/basinhopping_audit_seed_cap_fix/run_audit/audit.json",
}


def _load_rows() -> list[tuple[str, str, dict]]:
    rows = []
    for rel_path in _MULTI_FUNCTION_ARTIFACTS:
        data = json.loads((REPO_ROOT / rel_path).read_text())
        for function, row in data["functions"].items():
            rows.append((rel_path, function, row))
    for rel_path in _SINGLE_FUNCTION_ARTIFACTS:
        row = json.loads((REPO_ROOT / rel_path).read_text())
        rows.append((rel_path, row["config"]["function"], row))
    return rows


# --- Unit tests: _guard_margin_degeneracy itself --------------------------


def _verdict(verdict: Verdict, *, margin_degenerate: bool):
    control = [1e-11, 2e-11, 1.5e-11] if margin_degenerate else [1.0, 1.1, 0.9]
    treatment = [0.05, 0.1, 0.08] if margin_degenerate else [1.0, 0.95, 1.05]
    report = assess_margin_degeneracy(control, treatment)
    assert report.degenerate is margin_degenerate  # sanity on the fixture itself
    base = equivalence_verdict(treatment, control, metric="m", margin=1e-9, confidence=0.90)
    return dataclasses.replace(base, verdict=verdict, margin_degeneracy=report)


def test_degenerate_metric_is_withdrawn_to_inconclusive() -> None:
    verdict = _verdict(Verdict.HARMFUL, margin_degenerate=True)
    guarded, withdrawn = _guard_margin_degeneracy({"m": verdict})
    assert withdrawn == ["m"]
    assert guarded["m"].verdict is Verdict.INCONCLUSIVE
    assert "degenerate" in (guarded["m"].test or "").lower()
    # Nothing deleted: interval/p-value/margin survive the withdrawal.
    assert guarded["m"].ci_low == verdict.ci_low
    assert guarded["m"].ci_high == verdict.ci_high
    assert guarded["m"].margin == verdict.margin
    assert guarded["m"].p_value == verdict.p_value


def test_degenerate_contributes_or_harmful_loses_its_boundary_clearance_ratio() -> None:
    """A withdrawal to INCONCLUSIVE is no longer a superiority claim, so its
    boundary_clearance_ratio must not survive -- same invariant
    _guard_vacuous_comparison and _holm_correct already enforce."""
    verdict = dataclasses.replace(
        _verdict(Verdict.HARMFUL, margin_degenerate=True), boundary_clearance_ratio=3.2
    )
    guarded, _ = _guard_margin_degeneracy({"m": verdict})
    assert guarded["m"].boundary_clearance_ratio is None


def test_non_degenerate_metric_is_untouched() -> None:
    verdict = _verdict(Verdict.NULL, margin_degenerate=False)
    guarded, withdrawn = _guard_margin_degeneracy({"m": verdict})
    assert withdrawn == []
    assert guarded["m"] is verdict


def test_metric_with_no_margin_degeneracy_report_is_untouched() -> None:
    """A verdict built without going through arms.audit() (margin_degeneracy
    is None) must not crash the guard or be treated as degenerate."""
    verdict = equivalence_verdict([1.0, 1.1], [0.9, 1.0], metric="m", margin=1.0)
    assert verdict.margin_degeneracy is None
    guarded, withdrawn = _guard_margin_degeneracy({"m": verdict})
    assert withdrawn == []
    assert guarded["m"] is verdict


def test_mixed_family_only_withdraws_the_degenerate_metric() -> None:
    degenerate = _verdict(Verdict.HARMFUL, margin_degenerate=True)
    clean = _verdict(Verdict.CONTRIBUTES, margin_degenerate=False)
    guarded, withdrawn = _guard_margin_degeneracy({"bad": degenerate, "good": clean})
    assert withdrawn == ["bad"]
    assert guarded["bad"].verdict is Verdict.INCONCLUSIVE
    assert guarded["good"].verdict is Verdict.CONTRIBUTES  # untouched


# --- Live wiring: arms.audit() actually withdraws -------------------------


class _FrozenControlSearcher:
    """Control-arm restarts converge to (numerically) one point; treatment
    varies normally -- the exact shape of the incident this gate exists for."""

    restart_cost: int = 1

    def __init__(self, *, treatment: bool) -> None:
        self._treatment = treatment

    def search(self, problem, seed):
        rng = np.random.default_rng(seed)
        if self._treatment:
            value = float(rng.normal(10.0, 2.0))
        else:
            value = 1e-11 + float(rng.normal(0.0, 1e-12))
        return SearchOutcome(
            metrics={"quality": value}, evaluations_used=1, representation=f"c{seed}"
        )

    def select(self, outcomes):
        return max(outcomes, key=lambda outcome: outcome.metrics["quality"])


class _DegenerateControlScaffold:
    """A minimal Scaffold whose control arm is frozen and treatment arm is
    not -- exercised directly through arms.audit(), not through a
    calibration scaffold, since none of NullScaffold/OracleScaffold/
    WastefulScaffold produce mismatched arm distributions by construction."""

    name = "degenerate-control-test-scaffold"

    def unwrap(self):
        return _FrozenControlSearcher(treatment=False)

    def run(self, problem, seed):
        return _FrozenControlSearcher(treatment=True).search(problem, seed)


def test_audit_withdraws_a_genuinely_margin_degenerate_live_run() -> None:
    report = audit(
        _DegenerateControlScaffold(),
        problem=None,
        seeds=list(range(20)),
        margins={"quality": 1e-9},
        higher_is_better={"quality": True},
    )
    metric_verdict = report.per_metric["quality"]
    assert metric_verdict.margin_degeneracy is not None
    assert metric_verdict.margin_degeneracy.degenerate
    assert metric_verdict.verdict is Verdict.INCONCLUSIVE
    assert metric_verdict.test is not None and "degenerate" in metric_verdict.test.lower()
    assert any("MARGIN_DEGENERATE" in limitation for limitation in report.limitations)


# --- Retrospective validation: this branch's actual SPEC promise ----------


@pytest.mark.parametrize(
    "rel_path,function,row",
    [pytest.param(path, fn, row, id=f"{path}:{fn}") for path, fn, row in _load_rows()],
)
def test_gating_reproduces_decision_log_judgment(rel_path, function, row) -> None:
    """For every committed row, build the real MetricVerdict via
    equivalence_verdict + assess_margin_degeneracy from the row's own raw
    arm data, run it through _guard_margin_degeneracy, and check the
    resulting verdict against DECISION_LOG.md's established trust labels."""
    control = row["arms"]["control_objective"]
    treatment = row["arms"]["treatment_objective"]
    margin = row["config"]["margin"]

    base_verdict = equivalence_verdict(treatment, control, metric="objective", margin=margin)
    report = assess_margin_degeneracy(control, treatment)
    verdict = dataclasses.replace(base_verdict, margin_degeneracy=report)

    guarded, withdrawn = _guard_margin_degeneracy({"objective": verdict})

    is_untrusted_griewank = function == "griewank" and rel_path in _UNTRUSTED_GRIEWANK_FILES
    if is_untrusted_griewank:
        assert withdrawn == ["objective"], f"{rel_path}:{function} should have been gated"
        assert guarded["objective"].verdict is Verdict.INCONCLUSIVE
    else:
        assert withdrawn == [], (
            f"{rel_path}:{function} was gated but is trusted per DECISION_LOG.md"
        )
        assert (
            guarded["objective"].verdict is verdict.verdict
        )  # untouched, not just same enum value
