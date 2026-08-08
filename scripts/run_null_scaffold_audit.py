"""Run the null-scaffold audit against the DiscoveryAgent pipeline.

    python scripts/run_null_scaffold_audit.py --subset smoke --seeds 10

Writes JSON and CSV artifacts under `results/null_scaffold_audit/`. Every number
this project reports about the audit comes from these files (Article 13); none
is typed into prose by hand.

PRE-REGISTERED MARGINS
----------------------
The practical-equivalence margins below were fixed before the first sweep was
run and are not to be adjusted after seeing an interval. An audit whose margin
moves to fit its results is not an audit.

- `rmse`:                 5% of the test target's standard deviation.
- `symbolic_complexity`:  2 nodes.
- `exact_recovery`:       0.10 (10 percentage points).

The rmse margin is scaled by the problem rather than fixed absolutely because
the benchmark's targets span many orders of magnitude, so a single absolute
number would be vacuous on one equation and unattainable on another. It is
scaled by the *target*, not by either arm's error, so it remains a property of
the problem and cannot be influenced by the results.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.audit import NotSeparableError, Verdict, audit
from physics_discovery.audit.agent_adapter import (
    DiscoveryAgentScaffold,
    RediscoveryProblem,
)
from physics_discovery.data.feynman_loader import (
    generate_feynman_dataset,
    list_feynman_equations,
)

RMSE_MARGIN_FRACTION = 0.05
COMPLEXITY_MARGIN = 2.0
RECOVERY_MARGIN = 0.10

HIGHER_IS_BETTER = {
    "rmse": False,
    "symbolic_complexity": False,
    "exact_recovery": True,
}


def build_problem(equation_id: str, n_samples: int, seed: int) -> RediscoveryProblem:
    X, y, ground_truth = generate_feynman_dataset(equation_id, n_samples=n_samples, seed=seed)
    split = int(0.8 * len(y))
    return RediscoveryProblem(
        equation_id=equation_id,
        x_train=X[:split],
        y_train=y[:split],
        x_test=X[split:],
        y_test=y[split:],
        ground_truth=ground_truth,
    )


def margins_for(problem: RediscoveryProblem) -> dict[str, float]:
    target_spread = float(np.std(np.asarray(problem.y_test, dtype=float)))
    return {
        # A degenerate target (zero spread) would make the margin zero, which the
        # statistics layer rejects rather than silently treating as "any
        # difference matters". Fall back to an absolute floor so the audit
        # reports a verdict instead of an exception.
        "rmse": max(RMSE_MARGIN_FRACTION * target_spread, 1e-9),
        "symbolic_complexity": COMPLEXITY_MARGIN,
        "exact_recovery": RECOVERY_MARGIN,
    }


def write_artifacts(
    out: Path, config: dict[str, Any], reports: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> None:
    """Write the artifacts, overwriting whatever was there.

    Called after every problem, not once at the end. A long sweep that is
    interrupted -- and these run for hours -- would otherwise produce nothing at
    all, discarding every completed problem because one was still running.
    Partial results are worth strictly more than no results, provided the file
    says how far it got, which `problems_completed` does.
    """
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {**config, "problems_completed": len(reports)},
        "reports": reports,
    }
    (out / "audit.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if rows:
        with (out / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["smoke", "all"], default="smoke")
    parser.add_argument("--equations", nargs="*", default=None, help="Explicit equation ids.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "null_scaffold_audit")
    args = parser.parse_args()

    if args.equations:
        equation_ids = list(args.equations)
    else:
        entries = list_feynman_equations()
        equation_ids = [e["id"] for e in (entries[:8] if args.subset == "smoke" else entries)]

    seeds = list(range(args.seeds))
    scaffold = DiscoveryAgentScaffold(
        max_iters=args.max_iters,
        population_size=args.population_size,
        generations=args.generations,
    )

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    config: dict[str, Any] = {
        "seeds": len(seeds),
        "max_iters": args.max_iters,
        "population_size": args.population_size,
        "generations": args.generations,
        "n_samples": args.n_samples,
        "evaluations_per_fit": args.population_size * args.generations,
        "margins": {
            "rmse": f"{RMSE_MARGIN_FRACTION} * std(y_test)",
            "symbolic_complexity": COMPLEXITY_MARGIN,
            "exact_recovery": RECOVERY_MARGIN,
        },
    }

    for equation_id in equation_ids:
        problem = build_problem(equation_id, args.n_samples, seed=0)
        print(f"[audit] {equation_id}: {len(seeds)} seeds", flush=True)
        try:
            report = audit(
                scaffold,
                problem,
                seeds,
                margins=margins_for(problem),
                higher_is_better=HIGHER_IS_BETTER,
            )
        except NotSeparableError as exc:
            print(f"[audit] {equation_id}: NOT_SEPARABLE ({exc})", flush=True)
            rows.append(
                {"equation_id": equation_id, "metric": "-", "verdict": Verdict.NOT_SEPARABLE.value}
            )
            continue

        arms = report.arms
        print(
            f"        verdict={report.verdict.value} "
            f"identical={arms.identical_representation_rate:.0%} "
            f"restarts={arms.restarts_per_seed} "
            f"evals: treatment={arms.treatment_evaluations} control={arms.control_evaluations}",
            flush=True,
        )
        print(f"        degeneracy: {report.degeneracy.summary()}", flush=True)
        for metric, verdict in report.per_metric.items():
            print(
                f"        {metric:<20} {verdict.verdict.value:<13} "
                f"diff={verdict.observed_difference:+.4g} "
                f"CI=[{verdict.ci_low:+.4g}, {verdict.ci_high:+.4g}] "
                f"margin=+/-{verdict.margin:.4g} power={verdict.power:.2f}",
                flush=True,
            )
            rows.append(
                {
                    "equation_id": equation_id,
                    "metric": metric,
                    "verdict": verdict.verdict.value,
                    "observed_difference": verdict.observed_difference,
                    "ci_low": verdict.ci_low,
                    "ci_high": verdict.ci_high,
                    "margin": verdict.margin,
                    "power": verdict.power,
                    "n": verdict.n,
                    "higher_is_better": verdict.higher_is_better,
                }
            )

        reports.append(
            {
                "equation_id": equation_id,
                "scaffold": report.scaffold,
                "verdict": report.verdict.value,
                "identical_representation_rate": arms.identical_representation_rate,
                "degeneracy": dataclasses.asdict(report.degeneracy),
                "degenerate": report.degeneracy.degenerate,
                "restarts_per_seed": arms.restarts_per_seed,
                "treatment_evaluations": arms.treatment_evaluations,
                "control_evaluations": arms.control_evaluations,
                "seeds": arms.seeds,
                "per_metric": {m: dataclasses.asdict(v) for m, v in report.per_metric.items()},
                "treatment_representations": [o.representation for o in arms.treatment],
                "control_representations": [o.representation for o in arms.control],
                "limitations": report.limitations,
            }
        )
        write_artifacts(args.out, config, reports, rows)

    write_artifacts(args.out, config, reports, rows)

    verdict_counts: dict[str, int] = {}
    for report_row in reports:
        verdict_counts[report_row["verdict"]] = verdict_counts.get(report_row["verdict"], 0) + 1
    print(f"\n[audit] overall verdicts across {len(reports)} problems: {verdict_counts}")
    print(f"[audit] wrote {args.out / 'audit.json'} and {args.out / 'audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
