"""Re-run PR #16's audit with a domain-scaled stepsize, per STEPSIZE_SPEC.md.

    python plugins/basinhopping_audit/run_stepsize_experiment.py

Issue #17 / STEPSIZE_SPEC.md: holds everything from `run_audit.py` fixed
(functions, dimension, local minimizer, budget-matching procedure,
statistical procedure) except `stepsize`, which was a single shared
absolute value (scipy's default, 0.5) in PR #16 and is here set
proportionally to each function's domain width instead, using Rastrigin's
own PR #16 configuration as the fixed proportional standard:

    ratio = 0.5 / (Rastrigin's domain width, 10.24) ~= 0.048828125
    stepsize(function) = ratio * (function's domain width)

This reproduces STEPSIZE_SPEC.md's stated values exactly: Rastrigin ~=0.5
(unchanged, by construction), Ackley ~=3.20, Griewank ~=58.59.

Seeds are chosen disjoint from *both* of PR #16's seed blocks (its pilot
seeds 1000-1014 and its real-sweep seeds 0-59), not just disjoint from this
experiment's own pilot -- so nothing here can be read as reusing or
cherry-picking from the original run's randomness. Margins and seed counts
are re-derived from a fresh feasibility probe under the new stepsize, per
STEPSIZE_SPEC.md's explicit instruction not to reuse PR #16's margins (a
different stepsize changes both arms' variance).
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

OUT = REPO_ROOT / "results" / "basinhopping_audit_stepsize_scaling"

PILOT_NITER = 50  # unchanged from PR #16
RASTRIGIN_WIDTH = PROBLEM_SET["rastrigin"].bounds[0][1] - PROBLEM_SET["rastrigin"].bounds[0][0]
RASTRIGIN_STEPSIZE = 0.5  # PR #16's value, the fixed proportional standard
STEPSIZE_RATIO = RASTRIGIN_STEPSIZE / RASTRIGIN_WIDTH


def domain_scaled_stepsize(name: str) -> float:
    width = PROBLEM_SET[name].bounds[0][1] - PROBLEM_SET[name].bounds[0][0]
    return STEPSIZE_RATIO * width


# Disjoint from PR #16's pilot seeds (1000-1014) and real-sweep seeds
# (0 through at most 59, since MAX_SEEDS there was 60).
PILOT_SEEDS = list(range(2000, 2015))
REAL_SWEEP_SEED_OFFSET = 10_000

MARGIN_FRACTION = 0.25  # unchanged from PR #16
MIN_SEEDS = 20
MAX_SEEDS = 60
CEILING_RESTARTS_CAP = 10


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_reports: dict[str, dict] = {}

    print(
        f"[stepsize-experiment] ratio (from rastrigin, width={RASTRIGIN_WIDTH}, "
        f"stepsize={RASTRIGIN_STEPSIZE}) = {STEPSIZE_RATIO:.6f}",
        flush=True,
    )

    for name, problem in PROBLEM_SET.items():
        stepsize = domain_scaled_stepsize(name)
        print(
            f"[stepsize-experiment] === {name} (dim={DIMENSION}, stepsize={stepsize:.4g}) ===",
            flush=True,
        )
        base = LocalMinimizerRestart()
        scaffold = BasinhoppingScaffold(niter=PILOT_NITER, stepsize=stepsize)

        # 1. Fresh feasibility probe under the new stepsize.
        t0 = time.time()
        pilot_arms = run_arms(scaffold, problem, PILOT_SEEDS)
        control_values = np.array([o.metrics[METRIC] for o in pilot_arms.control])
        treatment_values = np.array([o.metrics[METRIC] for o in pilot_arms.treatment])
        diffs = control_values - treatment_values
        spread = float(diffs.std(ddof=1))
        control_spread = float(control_values.std(ddof=1))
        margin = max(MARGIN_FRACTION * control_spread, 1e-9)

        needed_n = required_sample_size(margin, spread, confidence=0.90, target_power=0.80)
        seed_count = MIN_SEEDS if needed_n is None else min(max(needed_n, MIN_SEEDS), MAX_SEEDS)
        print(
            f"[stepsize-experiment] {name}: probe spread={spread:.4g} "
            f"control_spread={control_spread:.4g} margin={margin:.4g} required_n={needed_n} "
            f"-> seed_count={seed_count} ({time.time() - t0:.1f}s)",
            flush=True,
        )

        # 2. Selection ceiling, same pilot seeds.
        ceiling_restarts = min(PILOT_NITER + 1, CEILING_RESTARTS_CAP)
        ceiling = selection_ceiling(
            base,
            problem,
            PILOT_SEEDS,
            metric=METRIC,
            restarts=ceiling_restarts,
            higher_is_better=False,
        )

        # 3. Real audit, fresh seeds disjoint from PR #16's blocks *and*
        # from this experiment's own pilot.
        real_seeds = list(range(REAL_SWEEP_SEED_OFFSET, REAL_SWEEP_SEED_OFFSET + seed_count))
        t0 = time.time()
        report = audit(
            scaffold,
            problem,
            real_seeds,
            margins={METRIC: margin},
            higher_is_better={METRIC: False},
        )
        elapsed = time.time() - t0
        verdict = report.per_metric[METRIC]
        print(
            f"[stepsize-experiment] {name}: verdict={report.verdict.value} "
            f"diff={verdict.observed_difference:+.4g} CI=[{verdict.ci_low:+.4g}, "
            f"{verdict.ci_high:+.4g}] margin={margin:.4g} power={verdict.power:.2f} "
            f"degeneracy={report.degeneracy.summary()} ({elapsed:.1f}s)",
            flush=True,
        )

        distance_to_optimum = [float(abs(v - problem.global_optimum)) for v in treatment_values]

        all_reports[name] = {
            "config": {
                "function": name,
                "dimension": DIMENSION,
                "bounds": problem.bounds,
                "global_optimum": problem.global_optimum,
                "local_method": LOCAL_METHOD,
                "stepsize": stepsize,
                "stepsize_ratio": STEPSIZE_RATIO,
                "niter": PILOT_NITER,
                "pilot_seeds": PILOT_SEEDS,
                "pilot_spread": spread,
                "pilot_control_spread": control_spread,
                "margin_fraction": MARGIN_FRACTION,
                "margin": margin,
                "required_n_for_80pct_power": needed_n,
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

    (OUT / "audit.json").write_text(
        json.dumps({"functions": all_reports}, indent=2, default=str), encoding="utf-8"
    )
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
        for name, r in all_reports.items():
            v = r["per_metric"][METRIC]
            writer.writerow(
                [
                    name,
                    r["config"]["stepsize"],
                    r["verdict"],
                    v["observed_difference"],
                    v["ci_low"],
                    v["ci_high"],
                    v["margin"],
                    v["power"],
                    v["n"],
                    r["ceiling"]["ceiling"],
                    r["ceiling"]["ceiling_upper"],
                    r["degeneracy"]["degenerate"],
                ]
            )

    print(f"\n[stepsize-experiment] wrote {OUT / 'audit.json'} and {OUT / 'audit.csv'}")
    for name, r in all_reports.items():
        print(
            f"[stepsize-experiment] {name}: {r['verdict']} (stepsize={r['config']['stepsize']:.4g})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
