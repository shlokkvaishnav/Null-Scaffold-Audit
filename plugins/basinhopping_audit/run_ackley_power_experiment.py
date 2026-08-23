"""Resolve Ackley's INCONCLUSIVE basinhopping verdict at adequate power, per
ACKLEY_POWER_SPEC.md (issue #19).

    python plugins/basinhopping_audit/run_ackley_power_experiment.py

PR #18 (issue #17) ran Ackley at `stepsize=3.2` (domain-scaled) but its own
feasibility probe called for `n~=135` while the shared runner script capped
the real sweep at `MAX_SEEDS=60` -- an avoidable power shortfall, not a
fundamental limit (Ackley's evaluations are as cheap as any other function
here). This re-runs only Ackley, same `stepsize=3.2`, with a *fresh*
feasibility probe (seeds disjoint from every prior block) and an
**uncapped** seed count -- whatever the fresh probe's own power calculation
recommends, not PR #18's `135` reused verbatim and not any hardcoded
ceiling.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from engine.audit.arms import audit, run_arms
from engine.audit.calibration import selection_ceiling
from engine.audit.statistics import required_sample_size
from plugins.basinhopping_audit.functions import DIMENSION, PROBLEM_SET
from plugins.basinhopping_audit.scaffold import BasinhoppingScaffold
from plugins.basinhopping_audit.searcher import LOCAL_METHOD, METRIC, LocalMinimizerRestart

OUT = REPO_ROOT / "results" / "basinhopping_audit_ackley_power"

FUNCTION = "ackley"
NITER = 50  # unchanged from PR #16/#18
STEPSIZE = 3.2  # PR #18's domain-scaled value for Ackley, unchanged by this branch
MARGIN_FRACTION = 0.25  # unchanged from PR #16/#18

# Disjoint from every prior block:
#   PR #16 (issue #15): pilot 1000-1014, real sweep 0-59
#   PR #18 (issue #17): pilot 2000-2014, real sweep 10000-10059
PILOT_SEEDS = list(range(3000, 3015))
REAL_SWEEP_SEED_OFFSET = 20_000
MIN_SEEDS = 20  # a floor, not a ceiling -- no MAX_SEEDS here, per SPEC.md

PRIOR_PROBE_ESTIMATE = 135
"""PR #18's `required_n_for_80pct_power` for Ackley, reported for
comparison only -- this branch computes and uses its own fresh estimate,
per ACKLEY_POWER_SPEC.md's explicit instruction not to reuse this number
verbatim."""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    problem = PROBLEM_SET[FUNCTION]
    base = LocalMinimizerRestart()
    scaffold = BasinhoppingScaffold(niter=NITER, stepsize=STEPSIZE)

    print(f"[ackley-power] === {FUNCTION} (dim={DIMENSION}, stepsize={STEPSIZE}) ===", flush=True)

    # 1. Fresh feasibility probe.
    t0 = time.time()
    pilot_arms = run_arms(scaffold, problem, PILOT_SEEDS)
    control_values = np.array([o.metrics[METRIC] for o in pilot_arms.control])
    treatment_values = np.array([o.metrics[METRIC] for o in pilot_arms.treatment])
    diffs = control_values - treatment_values
    spread = float(diffs.std(ddof=1))
    control_spread = float(control_values.std(ddof=1))
    margin = max(MARGIN_FRACTION * control_spread, 1e-9)

    needed_n = required_sample_size(margin, spread, confidence=0.90, target_power=0.80)
    seed_count = max(needed_n, MIN_SEEDS) if needed_n is not None else MIN_SEEDS
    print(
        f"[ackley-power] fresh probe: spread={spread:.4g} control_spread={control_spread:.4g} "
        f"margin={margin:.4g} required_n={needed_n} (PR #18's estimate was "
        f"{PRIOR_PROBE_ESTIMATE}) -> seed_count={seed_count} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # 2. Selection ceiling, re-measured (not assumed to carry over).
    ceiling_restarts = min(NITER + 1, 10)
    ceiling = selection_ceiling(
        base, problem, PILOT_SEEDS, metric=METRIC, restarts=ceiling_restarts, higher_is_better=False
    )
    print(f"[ackley-power] ceiling={ceiling['ceiling']:.4g} (upper {ceiling['ceiling_upper']:.4g})")

    # 3. Real audit, uncapped seed count, fresh disjoint seeds.
    real_seeds = list(range(REAL_SWEEP_SEED_OFFSET, REAL_SWEEP_SEED_OFFSET + seed_count))
    t0 = time.time()
    report = audit(
        scaffold, problem, real_seeds, margins={METRIC: margin}, higher_is_better={METRIC: False}
    )
    elapsed = time.time() - t0
    verdict = report.per_metric[METRIC]
    print(
        f"[ackley-power] verdict={report.verdict.value} diff={verdict.observed_difference:+.4g} "
        f"CI=[{verdict.ci_low:+.4g}, {verdict.ci_high:+.4g}] margin={margin:.4g} "
        f"power={verdict.power:.3f} n={seed_count} degeneracy={report.degeneracy.summary()} "
        f"({elapsed:.1f}s)",
        flush=True,
    )

    distance_to_optimum = [float(abs(v - problem.global_optimum)) for v in treatment_values]

    payload = {
        "config": {
            "function": FUNCTION,
            "dimension": DIMENSION,
            "bounds": problem.bounds,
            "global_optimum": problem.global_optimum,
            "local_method": LOCAL_METHOD,
            "stepsize": STEPSIZE,
            "niter": NITER,
            "pilot_seeds": PILOT_SEEDS,
            "pilot_spread": spread,
            "pilot_control_spread": control_spread,
            "margin_fraction": MARGIN_FRACTION,
            "margin": margin,
            "required_n_for_80pct_power": needed_n,
            "prior_probe_estimate_pr18": PRIOR_PROBE_ESTIMATE,
            "seed_count": seed_count,
            "seeds": real_seeds,
        },
        "ceiling": {**ceiling, "restarts_used": ceiling_restarts},
        "verdict": report.verdict.value,
        "per_metric": {METRIC: dataclasses.asdict(verdict)},
        "degeneracy": {
            **dataclasses.asdict(report.degeneracy),
            "degenerate": report.degeneracy.degenerate,
        },
        "arms": {
            "identical_representation_rate": report.arms.identical_representation_rate,
            "restarts_per_seed": report.arms.restarts_per_seed,
            "treatment_evaluations": report.arms.treatment_evaluations,
            "control_evaluations": report.arms.control_evaluations,
            "treatment_objective": [o.metrics[METRIC] for o in report.arms.treatment],
            "control_objective": [o.metrics[METRIC] for o in report.arms.control],
            "treatment_distance_to_global_optimum_mean": float(np.mean(distance_to_optimum)),
            "treatment_distance_to_global_optimum_min": float(np.min(distance_to_optimum)),
        },
        "limitations": report.limitations,
        "elapsed_sec": elapsed,
    }

    (OUT / "audit.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    with (OUT / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "function",
                "stepsize",
                "verdict",
                "observed_difference",
                "ci_low",
                "ci_high",
                "margin",
                "power",
                "n",
                "ceiling",
                "ceiling_upper",
                "degenerate",
            ]
        )
        writer.writerow(
            [
                FUNCTION,
                STEPSIZE,
                report.verdict.value,
                verdict.observed_difference,
                verdict.ci_low,
                verdict.ci_high,
                verdict.margin,
                verdict.power,
                seed_count,
                ceiling["ceiling"],
                ceiling["ceiling_upper"],
                report.degeneracy.degenerate,
            ]
        )

    print(f"\n[ackley-power] wrote {OUT / 'audit.json'} and {OUT / 'audit.csv'}")
    print(
        f"[ackley-power] final: {report.verdict.value} at n={seed_count}, power={verdict.power:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
