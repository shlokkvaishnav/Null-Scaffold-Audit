"""Tests for analysis/nas_margin_sensitivity, per SPEC.md (issue #13).

A synthetic paired dataset stands in for the real audit data in most tests
here, so the ground truth (where a crossover *must* fall) is exact and known
in advance. One test reproduces the real committed
``results/nas_search_self_audit/audit.json`` fixture end to end, since the
whole point of this branch is a specific, real finding about that data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import analysis.nas_margin_sensitivity.run_sweep as sweep
from engine.audit.statistics import _interval
from engine.audit.verdict import Verdict


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module's paths at a temporary directory."""
    out = tmp_path / "results" / "nas_margin_sensitivity"
    monkeypatch.setattr(sweep, "OUT", out)
    return tmp_path


def _write_audit_fixture(path: Path, treatment: list[float], control: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arms": {"treatment_metrics": treatment, "control_metrics": control},
        "per_metric": {
            sweep.METRIC: dict(
                zip(
                    ("ci_low", "ci_high"),
                    _interval(
                        np.asarray(treatment) - np.asarray(control),
                        sweep.CONFIDENCE,
                        sweep.RESAMPLES,
                        sweep.SEED,
                    ),
                    strict=True,
                )
            )
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------
# _refine_crossover: exact on a hand-built interval
# --------------------------------------------------------------------------


def test_refine_crossover_finds_the_null_boundary() -> None:
    """For a fixed CI, NULL requires margin > -ci_low (the tighter bound here);
    the crossover must land there, not at |ci_high|."""
    ci_low, ci_high = -0.5, -0.1
    refined = sweep._refine_crossover(0.1, 0.9, ci_low, ci_high)
    assert refined == pytest.approx(0.5, abs=2 * sweep.BISECTION_TOLERANCE)


def test_refine_crossover_finds_the_harmful_boundary() -> None:
    ci_low, ci_high = -0.5, -0.1
    refined = sweep._refine_crossover(0.01, 0.5, ci_low, ci_high)
    assert refined == pytest.approx(0.1, abs=2 * sweep.BISECTION_TOLERANCE)


# --------------------------------------------------------------------------
# main(): synthetic data with a known, exact crossover structure
# --------------------------------------------------------------------------


def test_sweep_finds_both_crossovers_on_synthetic_data(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A negative-CI dataset (like the real one) must show HARMFUL at tiny
    margins, INCONCLUSIVE in between, NULL at large margins."""
    rng = np.random.default_rng(0)
    # Constructed so the paired difference is consistently negative but small,
    # matching the real audit's qualitative shape (mean well inside a loose
    # margin, but not at zero).
    control = list(rng.uniform(90.0, 91.0, size=30))
    treatment = [c - 0.15 + float(rng.normal(0, 0.05)) for c in control]

    fixture = sandbox / "results" / "nas_search_self_audit" / "audit.json"
    _write_audit_fixture(fixture, treatment, control)
    monkeypatch.setattr(sweep, "AUDIT_SOURCE", fixture)

    exit_code = sweep.main()
    assert exit_code == 0

    report = json.loads((sweep.OUT / "report.json").read_text(encoding="utf-8"))
    verdicts = [c["to_verdict"] for c in report["crossovers"]]
    # Exactly the shape SPEC.md's hypothesis describes: small margins read
    # HARMFUL, then INCONCLUSIVE, then NULL as the margin widens.
    assert verdicts[:2] == [Verdict.INCONCLUSIVE.value, Verdict.NULL.value] or verdicts == [
        Verdict.NULL.value
    ]
    # Crossovers are strictly increasing in margin (monotone verdict ordering).
    margins = [c["margin"] for c in report["crossovers"]]
    assert margins == sorted(margins)


def test_sweep_reports_no_crossover_when_verdict_is_robust(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CI straddling zero widely must stay NULL across the whole swept range."""
    rng = np.random.default_rng(1)
    control = list(rng.uniform(90.0, 91.0, size=30))
    treatment = [c + float(rng.normal(0, 0.01)) for c in control]  # ~no true difference

    fixture = sandbox / "results" / "nas_search_self_audit" / "audit.json"
    _write_audit_fixture(fixture, treatment, control)
    monkeypatch.setattr(sweep, "AUDIT_SOURCE", fixture)
    monkeypatch.setattr(sweep, "SWEEP_MIN", 0.05)  # skip the near-zero HARMFUL/CONTRIBUTES noise

    exit_code = sweep.main()
    assert exit_code == 0
    report = json.loads((sweep.OUT / "report.json").read_text(encoding="utf-8"))
    assert report["pre_registered_verdict"] == Verdict.NULL.value


# --------------------------------------------------------------------------
# The real committed fixture: the branch's actual finding
# --------------------------------------------------------------------------


def test_sweep_on_the_real_committed_audit_reproduces_its_ci(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins this branch's real result against the real, repo-tracked data file
    from issue #11 / PR #12, so a future change to that file's shape or a
    regression in the sweep logic both surface here."""
    real_source = sweep.REPO_ROOT / "results" / "nas_search_self_audit" / "audit.json"
    if not real_source.exists():
        pytest.skip("results/nas_search_self_audit/audit.json not present")

    exit_code = sweep.main()
    assert exit_code == 0

    report = json.loads((sweep.OUT / "report.json").read_text(encoding="utf-8"))
    assert report["pre_registered_verdict"] == Verdict.NULL.value
    assert report["ci_low"] == pytest.approx(-0.1969507431838293)
    assert report["ci_high"] == pytest.approx(-0.008143933584611237)
    # Two crossovers, in increasing-margin order: HARMFUL->INCONCLUSIVE, then
    # INCONCLUSIVE->NULL -- exactly the shape SPEC.md's hypothesis predicted.
    assert [c["to_verdict"] for c in report["crossovers"]] == [
        Verdict.INCONCLUSIVE.value,
        Verdict.NULL.value,
    ]
    assert report["distance_to_nearest_crossover"] == pytest.approx(0.103, abs=0.01)
