"""Run this branch's actual experiment: audit RandomSearch against itself.

    NATS_BENCH_TSS_SIMPLE_PATH=/path/to/NATS-tss-v1_0-3ffb9-simple \
        python -m plugins.nas_search.run_self_audit

This is deliberately not folded into `scripts/run_null_scaffold_audit.py`:
that script's margins, metric names, and re-evaluation helpers
(`margins_for`, `HIGHER_IS_BETTER`, `test_rmse`) are the physics/symbolic-
regression plugin's -- rmse against a `y_test` split, an equation string --
and none of that applies to a tabular architecture lookup. Generalising that
script to be metric-agnostic is a real thing worth doing, but it is a change
to shared infrastructure, not something this issue's scope covers (SPEC.md,
"Confounds considered": this issue is about whether *this* plugin's
plumbing is trustworthy, not a refactor of the runner).

PRE-REGISTERED MARGIN
----------------------
``valid_accuracy``: 0.3 percentage points.

Chosen from a real feasibility probe on this problem and searcher (30 fake
seeds is not this program's convention; this margin was fixed *before* the
sweep this module's `main()` runs, matching that probe): a 20-seed null
calibration run (`IdentityRestartScaffold` vs. `RandomSearch`'s own bare
restarts, budget=50, restarts=3) measured a paired-difference spread of
0.315 percentage points, giving a minimum detectable effect of 0.143 points
at 30 seeds (alpha=0.05 one-sided, power=0.80) -- comfortably under a 0.3
point margin, so the design can resolve an effect at that margin if one
exists. Not adjusted after seeing this sweep's own results.

The declared paired-binary metric from SPEC.md ("exact-architecture-match
rate") is *not* run through `margins`/`paired_binary` here. That pathway
(`AUDIT_METHODOLOGY.md` §4.2's Tango score interval) tests whether two
*independent per-arm success rates* agree -- e.g. physics' `exact_recovery`,
"did this run recover the true equation," computed once per outcome with no
reference to the other arm. What SPEC.md asks for is structurally different:
whether the treatment and control arms picked the *same* architecture on a
given seed, which is a comparison between two outcomes, not a property of
either alone -- and `engine.audit.arms.ArmOutcomes.identical_representation_
rate` already computes exactly that, seed by seed, using the same string
comparison `AuditReport`'s vacuous-comparison guard is built on. Routing it
through a second margin-tested metric would need a metric shape
`engine.audit.arms.audit()` does not have (a metric that reads both arms'
outcomes together), which is a change to `engine/`, not something this
plugin should invent unilaterally. So it is reported here as what it already
is: a descriptive statistic alongside the verdict, per seed and in aggregate,
not a third Holm-corrected claim.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.audit.arms import run_arms
from engine.audit.calibration import selection_ceiling
from engine.discovery import discover_plugins
from engine.registry import PluginRegistry
from plugins.nas_search import IdentityRestartScaffold, RandomSearch
from plugins.nas_search.problem import NatsBenchProblem
from plugins.nas_search.searcher import METRIC

VALID_ACCURACY_MARGIN = 0.3  # percentage points; see module docstring
SEEDS = list(range(30))
BUDGET = 50
RESTARTS = 3
CEILING_SEEDS = list(range(10))
OUT = REPO_ROOT / "results" / "nas_search_self_audit"


def main() -> int:
    registry = PluginRegistry()
    loaded = discover_plugins(registry)
    if "nas_search" not in loaded:
        print(
            "[nas_search] plugin did not register -- is NATS_BENCH_TSS_SIMPLE_PATH set?",
            file=sys.stderr,
        )
        return 2

    source = registry.build_problem_source("nas_search")
    # cast: `build_problem` is declared -> Any on `NatsBenchTopologyProblemSource`
    # (see plugins/nas_search/problem.py) precisely so this module can use the
    # real return shape instead of the physics plugin's `AuditProblem`.
    problem = cast(NatsBenchProblem, source.build_problem("cifar10-topology", n_samples=0, seed=0))

    base = RandomSearch(budget=BUDGET)
    scaffold = IdentityRestartScaffold(base=base, restarts=RESTARTS)

    print(
        f"[nas_search] selection ceiling: seeds={len(CEILING_SEEDS)} restarts={RESTARTS} "
        f"budget={BUDGET}",
        flush=True,
    )
    t0 = time.time()
    ceiling = selection_ceiling(
        base,
        problem,
        CEILING_SEEDS,
        metric=METRIC,
        restarts=RESTARTS,
        higher_is_better=True,
    )
    ceiling["elapsed_sec"] = time.time() - t0
    ceiling["margin"] = VALID_ACCURACY_MARGIN
    print(f"[nas_search]   ceiling={ceiling['ceiling']:.4f} ({ceiling['elapsed_sec']:.1f}s)")

    print(f"[nas_search] audit: seeds={len(SEEDS)} restarts={RESTARTS} budget={BUDGET}", flush=True)
    t0 = time.time()
    arms = run_arms(scaffold, problem, SEEDS)
    from engine.audit.degeneracy import assess_degeneracy
    from engine.audit.statistics import equivalence_verdict

    per_metric = {
        METRIC: equivalence_verdict(
            [o.metrics[METRIC] for o in arms.treatment],
            [o.metrics[METRIC] for o in arms.control],
            metric=METRIC,
            margin=VALID_ACCURACY_MARGIN,
            higher_is_better=True,
            confidence=0.90,
        )
    }
    degeneracy = assess_degeneracy([o.intermediate_representations for o in arms.treatment])
    elapsed = time.time() - t0

    verdict = per_metric[METRIC].verdict
    print(
        f"[nas_search]   verdict={verdict.value} elapsed={elapsed:.1f}s "
        f"identical_representation_rate={arms.identical_representation_rate:.3f} "
        f"degeneracy={degeneracy.summary()}"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    treatment_representations = [o.representation for o in arms.treatment]
    control_representations = [o.representation for o in arms.control]
    payload = {
        "config": {
            "domain": "nas_search",
            "problem": "cifar10-topology",
            "scaffold": scaffold.name,
            "searcher": "RandomSearch",
            "budget_per_search": BUDGET,
            "restarts": RESTARTS,
            "seeds": len(SEEDS),
            "margin": {METRIC: VALID_ACCURACY_MARGIN},
            "confidence": 0.90,
            "dataset": problem.dataset,
            "hp": problem.hp,
            "num_archs": problem.num_archs,
        },
        "ceiling": ceiling,
        "verdict": verdict.value,
        "per_metric": {METRIC: dataclasses.asdict(per_metric[METRIC])},
        "degeneracy": dataclasses.asdict(degeneracy),
        "arms": {
            "identical_representation_rate": arms.identical_representation_rate,
            "restarts_per_seed": arms.restarts_per_seed,
            "treatment_evaluations": arms.treatment_evaluations,
            "control_evaluations": arms.control_evaluations,
            "seeds": arms.seeds,
            "treatment_representations": treatment_representations,
            "control_representations": control_representations,
            "treatment_metrics": [o.metrics[METRIC] for o in arms.treatment],
            "control_metrics": [o.metrics[METRIC] for o in arms.control],
        },
        "elapsed_sec": elapsed,
    }
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    with (OUT / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "treatment", "control", "identical"])
        for seed, t_rep, c_rep in zip(
            arms.seeds,
            treatment_representations,
            control_representations,
            strict=True,
        ):
            writer.writerow([seed, t_rep, c_rep, t_rep == c_rep])

    print(f"[nas_search] wrote {OUT / 'audit.json'} and {OUT / 'audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
