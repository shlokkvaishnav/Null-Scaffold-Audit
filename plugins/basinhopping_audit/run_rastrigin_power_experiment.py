"""Verify Rastrigin's CONTRIBUTES verdict survives the seed-cap fix applied
to Ackley (issue #19 / PR #20), per RASTRIGIN_POWER_SPEC.md (issue #21).

    python plugins/basinhopping_audit/run_rastrigin_power_experiment.py

PR #18 (issue #17) ran Rastrigin at `stepsize=0.5` (its own domain-scaled
value, unchanged from PR #16) but its own feasibility probe called for
`n~=181` while the shared runner script capped the real sweep at
`MAX_SEEDS=60` -- the same defect issue #19 found and fixed for Ackley,
where the fix changed the verdict (`INCONCLUSIVE` -> `HARMFUL`). This
re-runs only Rastrigin, same `stepsize=0.5`, with a *fresh* feasibility
probe (seeds disjoint from every prior block in this line of research) and
an **uncapped** seed count -- mirroring
`run_ackley_power_experiment.py` exactly, substituting Rastrigin for
Ackley.
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

OUT = REPO_ROOT / "results" / "basinhopping_audit_rastrigin_power"

FUNCTION = "rastrigin"
NITER = 50  # unchanged from PR #16/#18
STEPSIZE = 0.5  # Rastrigin's domain-scaled value (== PR #16's original), unchanged
MARGIN_FRACTION = 0.25  # unchanged from PR #16/#18

# Disjoint from every prior block in this line of research:
#   PR #16 (issue #15): pilot 1000-1014, real sweep 0-59
#   PR #18 (issue #17): pilot 2000-2014, real sweep 10000-10059
#   PR #20 (issue #19): pilot 3000-3014, real sweep 20000-21970
PILOT_SEEDS = list(range(4000, 4015))
REAL_SWEEP_SEED_OFFSET = 30_000
MIN_SEEDS = 20  # a floor, not a ceiling -- no MAX_SEEDS here, per SPEC.md

PRIOR_PROBE_ESTIMATE = 181
"""PR #18's `required_n_for_80pct_power` for Rastrigin, reported for
comparison only -- this branch computes and uses its own fresh estimate,
per RASTRIGIN_POWER_SPEC.md's explicit instruction not to reuse this
number verbatim."""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    problem = PROBLEM_SET[FUNCTION]
    base = LocalMinimizerRestart()
    scaffold = BasinhoppingScaffold(niter=NITER, stepsize=STEPSIZE)

    print(
        f"[rastrigin-power] === {FUNCTION} (dim={DIMENSION}, stepsize={STEPSIZE}) ===", flush=True
    )

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
        f"[rastrigin-power] fresh probe: spread={spread:.4g} control_spread={control_spread:.4g} "
        f"margin={margin:.4g} required_n={needed_n} (PR #18's estimate was "
        f"{PRIOR_PROBE_ESTIMATE}) -> seed_count={seed_count} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # 2. Selection ceiling, re-measured (not assumed to carry over).
    ceiling_restarts = min(NITER + 1, 10)
    ceiling = selection_ceiling(
        base, problem, PILOT_SEEDS, metric=METRIC, restarts=ceiling_restarts, higher_is_better=False
    )
    print(
        f"[rastrigin-power] ceiling={ceiling['ceiling']:.4g} (upper {ceiling['ceiling_upper']:.4g})"
    )

    # 3. Real audit, uncapped seed count, fresh disjoint seeds.
    real_seeds = list(range(REAL_SWEEP_SEED_OFFSET, REAL_SWEEP_SEED_OFFSET + seed_count))
    t0 = time.time()
    report = audit(
        scaffold, problem, real_seeds, margins={METRIC: margin}, higher_is_better={METRIC: False}
    )
    elapsed = time.time() - t0
    verdict = report.per_metric[METRIC]
    print(
        f"[rastrigin-power] verdict={report.verdict.value} diff={verdict.observed_difference:+.4g} "
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

    print(f"\n[rastrigin-power] wrote {OUT / 'audit.json'} and {OUT / 'audit.csv'}")
    print(
        f"[rastrigin-power] final: {report.verdict.value} at n={seed_count}, power={verdict.power:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
