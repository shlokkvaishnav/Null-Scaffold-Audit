"""Tests for the cross-arm margin degeneracy check (issue #27,
``engine/audit/MARGIN_DEGENERACY_SPEC.md``).

The unit tests pin the mechanism (`assess_margin_degeneracy` itself). The
retrospective tests are the actual validation this branch's SPEC promised:
every already-committed `plugins/basinhopping_audit/` result, loaded from
its real `audit.json` artifact -- no new searcher/scaffold runs -- must
separate into exactly the two known-bad Griewank readings (degenerate) and
every other already-trusted row (not degenerate), including Griewank's own
domain-scaled-stepsize `NULL` reading. This is the confirmation, or
refutation, of the SPEC's Alternative hypothesis, not a demonstration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine.audit import MarginDegeneracyReport, SearchOutcome, assess_margin_degeneracy, audit

REPO_ROOT = Path(__file__).resolve().parents[1]

# (path relative to repo root, whether every function row in this file is
# already established, per DECISION_LOG.md, as trustworthy) -- Griewank rows
# specifically are singled out per-row below, since a file can hold both a
# trusted and an untrusted Griewank reading depending on which script and
# stepsize configuration produced it.
#
# The multi-function artifacts (PRs #16/#18/#26) nest rows under
# "functions"; the single-function power-experiment artifacts (PRs #20/#22)
# are the row itself at the top level -- both schemas are covered by
# _load_rows() below, per this SPEC's own stated validation scope
# (MARGIN_DEGENERACY_SPEC.md's Experimental design: "PRs #16, #18, #20,
# #22, and #26"; a reviewer pass on this branch caught the first version
# silently covering only #16/#18/#26).
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

# Per DECISION_LOG.md: only run_audit.py's unscaled-stepsize=0.5 Griewank
# rows (both the PR #16 and issue #25 readings) are flagged untrustworthy.
# run_stepsize_experiment.py's domain-scaled-stepsize Griewank row is
# trusted (stable NULL across both readings, consistent with issue #17).
# The two power-experiment artifacts (#20/#22) are Ackley/Rastrigin only --
# no Griewank row exists in either -- so nothing here needs to be untrusted.
_UNTRUSTED_GRIEWANK_FILES = {
    "results/basinhopping_audit/audit.json",
    "results/basinhopping_audit_seed_cap_fix/run_audit/audit.json",
}


def _load_rows() -> list[tuple[str, str, dict]]:
    """(artifact path, function name, row) for every function in every artifact."""
    rows = []
    for rel_path in _MULTI_FUNCTION_ARTIFACTS:
        data = json.loads((REPO_ROOT / rel_path).read_text())
        for function, row in data["functions"].items():
            rows.append((rel_path, function, row))
    for rel_path in _SINGLE_FUNCTION_ARTIFACTS:
        row = json.loads((REPO_ROOT / rel_path).read_text())
        rows.append((rel_path, row["config"]["function"], row))
    return rows


# --- Unit tests: the mechanism itself -------------------------------------


def test_frozen_control_beside_exploring_treatment_is_degenerate() -> None:
    """The exact shape of the incident this check exists to catch: control
    arm converged to (numerically) one point, treatment arm still varying."""
    control = [1e-11, 2e-11, 1.5e-11, 3e-11, 1.8e-11]
    treatment = [0.05, 0.12, 0.03, 0.20, 0.08]
    report = assess_margin_degeneracy(control, treatment)
    assert report.assessed
    assert report.degenerate
    assert report.ratio < 1e-6


def test_comparable_spreads_are_not_degenerate() -> None:
    """Both arms varying at the same scale -- an ordinary, trustworthy row."""
    control = [1.0, 1.2, 0.9, 1.1, 1.05]
    treatment = [0.8, 0.95, 1.1, 0.85, 1.0]
    report = assess_margin_degeneracy(control, treatment)
    assert report.assessed
    assert not report.degenerate


def test_both_arms_frozen_together_is_not_this_mechanism() -> None:
    """Both arms converged to the same point is `_guard_vacuous_comparison`'s
    structural failure, not this one -- the ratio (0/0) is undefined, so this
    check must not claim it either way."""
    control = [1e-11, 1e-11, 1e-11]
    treatment = [1e-11, 1e-11, 1e-11]
    report = assess_margin_degeneracy(control, treatment)
    assert not report.assessed
    assert not report.degenerate


def test_fewer_than_two_observations_is_unassessed() -> None:
    report = assess_margin_degeneracy([1e-11], [0.05, 0.1])
    assert not report.assessed
    assert not report.degenerate


def test_summary_reports_the_evidence() -> None:
    report = assess_margin_degeneracy([1e-11, 2e-11, 1.5e-11], [0.05, 0.1, 0.08])
    assert "MARGIN_DEGENERATE" in report.summary()
    assert not assess_margin_degeneracy([1.0, 1.1, 0.9], [1.0, 0.95, 1.05]).degenerate


def test_default_report_is_unassessed_not_a_silent_false() -> None:
    """An unassessed report and a confidently-not-degenerate report both
    return ``degenerate == False`` -- ``assessed`` is what distinguishes
    them, the same way ``DegeneracyReport`` works. A caller that checks only
    ``.degenerate`` without ``.assessed`` gets the conservative answer."""
    assert MarginDegeneracyReport(assessed=False).degenerate is False


# --- Retrospective validation: this branch's actual SPEC promise ----------


@pytest.mark.parametrize(
    "rel_path,function,row",
    [pytest.param(path, fn, row, id=f"{path}:{fn}") for path, fn, row in _load_rows()],
)
def test_margin_degeneracy_recomputed_from_committed_artifacts(rel_path, function, row) -> None:
    """Recompute margin degeneracy from each committed audit.json's raw
    per-seed arm values (`arms.control_objective`/`treatment_objective`),
    the same data `arms.audit()` has when it computes this live, and check
    it against this project's own established trust labels.

    Currently a single "objective" metric per row in every basinhopping
    artifact -- iterates in case a future artifact reports more than one.
    """
    control = row["arms"]["control_objective"]
    treatment = row["arms"]["treatment_objective"]
    report = assess_margin_degeneracy(control, treatment)

    is_untrusted_griewank = function == "griewank" and rel_path in _UNTRUSTED_GRIEWANK_FILES

    assert report.assessed, (
        f"{rel_path}:{function} was not assessed at all (a real committed row should "
        "always have >=2 observations per arm and nonzero treatment spread)"
    )
    if is_untrusted_griewank:
        assert report.degenerate, (
            f"{rel_path}:{function} is one of the two rows DECISION_LOG.md already "
            f"flags untrustworthy, but the check did not fire (ratio={report.ratio:.3g})"
        )
    else:
        assert not report.degenerate, (
            f"{rel_path}:{function} is trusted per DECISION_LOG.md, but the check "
            f"false-positived (ratio={report.ratio:.3g}) -- see this branch's SPEC "
            "Results for why this would refute the Alternative hypothesis"
        )


# --- Live wiring: arms.audit() actually attaches the report -------------


class _FakeSearcher:
    """Minimal BaseSearcher: quality varies by seed, nothing domain-specific."""

    restart_cost: int = 1

    def search(self, problem, seed):  # matches BaseSearcher's Protocol
        rng = np.random.default_rng(seed)
        return SearchOutcome(
            metrics={"quality": float(rng.normal(10.0, 2.0))},
            evaluations_used=1,
            representation=f"candidate-{seed}",
        )

    def select(self, outcomes):
        return max(outcomes, key=lambda outcome: outcome.metrics["quality"])


def test_audit_attaches_an_assessed_margin_degeneracy_report() -> None:
    """End-to-end through arms.audit(), not just the retrospective JSON
    replay above -- confirms the wiring in arms.py, not only the mechanism
    in margin_degeneracy.py."""
    from engine.audit import NullScaffold

    scaffold = NullScaffold(base=_FakeSearcher(), restarts=5)
    report = audit(
        scaffold,
        problem=None,
        seeds=list(range(20)),
        margins={"quality": 1.0},
        higher_is_better={"quality": True},
    )
    metric_verdict = report.per_metric["quality"]
    assert metric_verdict.margin_degeneracy is not None
    assert metric_verdict.margin_degeneracy.assessed
    # Both arms drawn from the same distribution here -- not the degenerate
    # shape at all, which is exactly what this asserts: the wiring reports
    # real evidence, not a hardcoded flag.
    assert not metric_verdict.margin_degeneracy.degenerate


def test_untrusted_and_trusted_griewank_ratios_are_orders_of_magnitude_apart() -> None:
    """The actual separation this branch's threshold sits inside of --
    documented as a number, not just a pass/fail, so a future reader does
    not have to re-derive it from the artifacts to trust the threshold."""
    ratios = {}
    for rel_path, function, row in _load_rows():
        if function != "griewank":
            continue
        report = assess_margin_degeneracy(
            row["arms"]["control_objective"], row["arms"]["treatment_objective"]
        )
        ratios[(rel_path, function)] = report.ratio

    untrusted = [r for (path, _fn), r in ratios.items() if path in _UNTRUSTED_GRIEWANK_FILES]
    trusted = [r for (path, _fn), r in ratios.items() if path not in _UNTRUSTED_GRIEWANK_FILES]
    assert len(untrusted) == 2
    assert len(trusted) == 2
    assert max(untrusted) < 1e-8
    assert min(trusted) > 0.2
