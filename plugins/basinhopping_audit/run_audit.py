"""Run this branch's actual experiment: audit basinhopping against itself, per SPEC.md.

    python plugins/basinhopping_audit/run_audit.py

For each function in `functions.PROBLEM_SET` (rastrigin, ackley, griewank):

1. **Feasibility probe** -- a pilot sweep at `PILOT_NITER` iterations, using
   seeds disjoint from the final sweep's, to estimate the paired-difference
   spread. From that spread: a pre-registered margin (`MARGIN_FRACTION` of
   the control arm's single-restart standard deviation -- a difference
   smaller than a quarter of ordinary restart-to-restart variability is not
   practically interesting) and the seed count `engine.audit.statistics.
   required_sample_size` says is needed for 80% power at that margin,
   floored at `MIN_SEEDS` (no upper cap -- see issue #25).
2. **Selection ceiling** (`engine.audit.calibration.selection_ceiling`) on
   the same pilot seeds, reported per SPEC.md's requirement to check rather
   than assume it.
3. **The real audit** (`engine.audit.arms.audit`), on a fresh, disjoint
   block of seeds at the niter/margin/seed-count the probe determined.

Margin, niter, and seed count are therefore chosen from the probe alone,
never from the direction or size of the real sweep's own result -- the real
sweep's seeds never overlap the probe's, so nothing about the final verdict
could have leaked into the choices that produced it.

Until issue #25, this script capped the real sweep's seed count at
`MAX_SEEDS=60` regardless of what the feasibility probe recommended --
confirmed (issue #19) to be able to change a verdict. That cap is removed;
see `SEED_CAP_FIX_SPEC.md` for the fix and a confirmatory re-run of every
row this script originally under-powered. The historical
`results/basinhopping_audit/audit.json` committed by PR #16 is
deliberately left untouched by that re-run (its own numbers are quoted in
`SPEC.md`'s Results section) -- running this corrected script for real
produces a new, larger-`n` result at fresh seeds, not a silent rewrite of
what PR #16 reported.
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
from plugins.basinhopping_audit.scaffold import DEFAULT_STEPSIZE, BasinhoppingScaffold
from plugins.basinhopping_audit.searcher import LOCAL_METHOD, METRIC, LocalMinimizerRestart

OUT = REPO_ROOT / "results" / "basinhopping_audit"

PILOT_NITER = 50
"""Iterations per arm for the feasibility probe and the real sweep alike --
decided once, before any results existed, and held fixed across both so the
probe's spread estimate applies to the sweep it informs."""

PILOT_SEEDS = list(range(7000, 7015))
"""15 seeds, disjoint from the real sweep's and from every other pilot
block used anywhere in plugins/basinhopping_audit/ (issue #25,
SEED_CAP_FIX_SPEC.md -- see that file for the full disjointness ledger).
Moved here from range(1000, 1015) when the MAX_SEEDS cap below was
removed: SEED_CAP_FIX_SPEC.md's design calls for a wholly fresh,
independently-drawn confirmation rather than extending the original
capped run's seed range."""
MARGIN_FRACTION = 0.25
"""Practical-equivalence margin, as a fraction of the control arm's single-
restart standard deviation across the pilot's seeds. A difference smaller
than a quarter of ordinary restart-to-restart variability is treated as not
practically interesting -- pre-registered as a fixed convention here, not
tuned per function after seeing an effect."""

MIN_SEEDS = 20
# No MAX_SEEDS: removed by issue #25 (SEED_CAP_FIX_SPEC.md). A silent
# ceiling here truncated the real sweep below what the feasibility probe's
# own power calculation required -- confirmed, empirically, to be able to
# change a verdict (issue #19's Ackley reversal). See
# run_ackley_power_experiment.py / run_rastrigin_power_experiment.py for
# the pattern this now matches.
REAL_SWEEP_SEED_OFFSET = 40_000
"""Where the real sweep's seeds start. Was `range(seed_count)` (i.e.
offset 0) before issue #25; moved off zero for the same reason PILOT_SEEDS
moved -- a fresh, disjoint confirmation, not an extension of the original
capped run's own seed range."""
CEILING_RESTARTS_CAP = 10
"""selection_ceiling's `restarts` -- capped well below niter+1 (which would
be 51): the ceiling only needs enough restarts to see whether there is a
selection gap at all, and its cost is `restarts` searches per seed, so this
keeps it a small fraction of the real sweep's cost as SPEC.md expects."""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_reports: dict[str, dict] = {}

    for name, problem in PROBLEM_SET.items():
        print(f"[basinhopping] === {name} (dim={DIMENSION}) ===", flush=True)
        base = LocalMinimizerRestart()
        scaffold = BasinhoppingScaffold(niter=PILOT_NITER)

        # 1. Feasibility probe.
        t0 = time.time()
        pilot_arms = run_arms(scaffold, problem, PILOT_SEEDS)
        control_values = np.array([o.metrics[METRIC] for o in pilot_arms.control])
        treatment_values = np.array([o.metrics[METRIC] for o in pilot_arms.treatment])
        diffs = control_values - treatment_values  # positive = treatment (lower) is better
        spread = float(diffs.std(ddof=1))
        control_spread = float(control_values.std(ddof=1))
        margin = max(MARGIN_FRACTION * control_spread, 1e-9)

        needed_n = required_sample_size(margin, spread, confidence=0.90, target_power=0.80)
        seed_count = MIN_SEEDS if needed_n is None else max(needed_n, MIN_SEEDS)
        print(
            f"[basinhopping] {name}: probe spread={spread:.4g} control_spread={control_spread:.4g} "
            f"margin={margin:.4g} required_n={needed_n} -> seed_count={seed_count} "
            f"({time.time() - t0:.1f}s)",
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
        print(
            f"[basinhopping] {name}: ceiling={ceiling['ceiling']:.4g} "
            f"(upper {ceiling['ceiling_upper']:.4g}), margin={margin:.4g}",
            flush=True,
        )

        # 3. Real audit, fresh disjoint seeds.
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
            f"[basinhopping] {name}: verdict={report.verdict.value} "
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
                "stepsize": DEFAULT_STEPSIZE,
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
                "degenerate": report.degeneracy.degenerate,  # a property, not a field
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

    print(f"\n[basinhopping] wrote {OUT / 'audit.json'} and {OUT / 'audit.csv'}")
    for name, r in all_reports.items():
        print(f"[basinhopping] {name}: {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
