"""Measure, per problem, the most that better selection could possibly buy.

    python scripts/measure_selection_ceiling.py \
        --domain physics \
        --scaffold plugins.physics.audit_adapter:DiscoveryAgentScaffold \
        --subset all --seeds 10 --workers 4

A wrapper spending its budget on the same searcher can beat the restart baseline
in exactly two ways: by selecting better among the candidates it generated, or by
generating better ones. This measures the ceiling on the first. No rule selects
better than taking the best candidate by the metric being measured, so the gap
between what the pipeline's own rule picks and the best available is an upper
bound -- approached by no real wrapper and exceeded by none.

Where that ceiling sits below the pre-registered margin, every wrapper that only
reorders or filters what the searcher produced is null on that problem *by
construction*, and no number of seeds will change it. Establishing that costs one
arm and a handful of seeds. Establishing it the other way costs a full two-arm
sweep per problem and returns `INCONCLUSIVE`, which reads as a failure of
evidence rather than what it is: a property of the design.

This runs one arm, so it is roughly half the cost of the audit whose results it
qualifies, and it should be run *first*.

The margins come from `run_null_scaffold_audit` rather than being restated here.
They are pre-registered, and a second copy is a second thing to drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_null_scaffold_audit import (
    HIGHER_IS_BETTER,
    default_registry,
    load_scaffold,
    margins_for,
)

from engine.audit.calibration import selection_ceiling

CEILING_METRIC = "rmse"


def _status(result: dict[str, float], margin: float) -> str:
    """closed / open / undetermined, from the interval rather than the mean."""
    lower = 2.0 * result["ceiling"] - result["ceiling_upper"]
    if result["ceiling_upper"] < margin:
        return "closed"
    if lower > margin:
        return "open"
    return "undetermined"


@dataclass(frozen=True)
class _Probe:
    """One problem's ceiling, as a picklable callable.

    Problems are independent and vary a lot in cost, so the parallelism is at
    this level rather than inside `selection_ceiling`. A process pool pickles
    whatever it is handed, which rules out a closure.

    The problem is rebuilt inside the worker rather than shipped to it: the
    arrays are large, and rebuilding from `(problem_id, n_samples, seed)` is both
    cheaper to send and exactly what the audit runner does.
    """

    domain: str
    scaffold_path: str
    n_samples: int
    seeds: tuple[int, ...]
    restarts: int
    population_size: int
    generations: int
    refit: bool = False

    def __call__(self, problem_id: str) -> dict[str, Any]:
        source = default_registry().build_problem_source(self.domain)
        problem = source.build_problem(problem_id, n_samples=self.n_samples, seed=0)
        base = load_scaffold(
            self.scaffold_path,
            max_iters=self.restarts,
            population_size=self.population_size,
            generations=self.generations,
        ).unwrap()
        if self.refit:
            # Imported here rather than at module scope: supplying a searcher with an
            # inner optimiser it lacks is one domain's business, and this script names
            # no domain anywhere else.
            from plugins.physics.audit_adapter import RefittingRestartSearcher

            base = RefittingRestartSearcher(
                population_size=self.population_size, generations=self.generations
            )

        result = selection_ceiling(
            base,
            problem,
            list(self.seeds),
            metric=CEILING_METRIC,
            restarts=self.restarts,
            higher_is_better=HIGHER_IS_BETTER[CEILING_METRIC],
        )
        margin = margins_for(problem)[CEILING_METRIC]
        return {
            "domain": self.domain,
            "problem": problem_id,
            "metric": CEILING_METRIC,
            "ceiling": result["ceiling"],
            "ceiling_sd": result["ceiling_sd"],
            "margin": margin,
            # How much of the margin the best possible selection rule could use
            # up. Below 1.0, no selection-only wrapper can reach CONTRIBUTES.
            "ratio": result["ceiling"] / margin if margin else float("inf"),
            "ceiling_upper": result["ceiling_upper"],
            "upper_ratio": result["ceiling_upper"] / margin if margin else float("inf"),
            # Three-way, because "we could not establish closure" is not closure.
            # `closed` needs the upper bound under the margin; `open` needs the lower
            # bound over it; anything straddling is undetermined and wants more seeds.
            "status": _status(result, margin),
            "metric_spread": result["metric_spread"],
            "spread_ratio": result["metric_spread"] / margin if margin else float("inf"),
            # Why a low ceiling is low. "tied" means the restarts landed in the
            # same place, so there was nothing to select between -- a fact about
            # the problem. "faithful" means they differed and the pipeline's own
            # rule already picked the best, which is a fact about the selection
            # signal and the stronger of the two findings.
            "null_because": (
                "reachable"
                if result["ceiling"] > margin
                else ("tied" if result["metric_spread"] < 0.5 * margin else "faithful")
            ),
            "seeds": int(result["seeds"]),
            "restarts": int(result["restarts"]),
        }


def write_artifacts(out: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write both artifacts, overwriting whatever was there.

    Called after every problem rather than once at the end, for the reason
    `run_null_scaffold_audit.write_artifacts` gives: a partial result that says
    how far it got is worth strictly more than no result, and these runs are
    long enough that being interrupted is the normal case rather than the
    exceptional one. `problems_completed` is what says how far it got.
    """
    out.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row["ratio"], reverse=True)
    payload = {"config": {**config, "problems_completed": len(ordered)}, "rows": ordered}
    (out / "ceiling.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if ordered:
        with (out / "ceiling.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ordered[0].keys()))
            writer.writeheader()
            writer.writerows(ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument(
        "--scaffold",
        required=True,
        help="module:Attribute whose unwrap() yields the primitive to measure.",
    )
    parser.add_argument("--subset", choices=["smoke", "all"], default="all")
    parser.add_argument("--problems", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--refit-constants",
        action="store_true",
        help="Fit each candidate's constants by least squares on the TRAIN split "
        "before measuring. Tests whether a near-zero ceiling reflects the problem or "
        "the searcher's lack of an inner optimiser.",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "selection_ceiling")
    args = parser.parse_args()

    registry = default_registry()
    try:
        source = registry.build_problem_source(args.domain)
    except KeyError as exc:
        print(f"[ceiling] {exc}", file=sys.stderr)
        return 2

    available = list(source.list_problems())
    if args.problems:
        unknown = [p for p in args.problems if p not in available]
        if unknown:
            print(f"[ceiling] {args.domain} has no problem(s) {unknown}", file=sys.stderr)
            return 2
        problem_ids = list(args.problems)
    elif args.subset == "smoke":
        problem_ids = available[:8]
    else:
        problem_ids = available

    probe = _Probe(
        domain=args.domain,
        scaffold_path=args.scaffold,
        n_samples=args.n_samples,
        seeds=tuple(range(args.seeds)),
        restarts=args.restarts,
        population_size=args.population_size,
        generations=args.generations,
        refit=args.refit_constants,
    )

    print(
        f"[ceiling] domain={args.domain} problems={len(problem_ids)} refit={args.refit_constants} "
        f"seeds={args.seeds} restarts={args.restarts} workers={args.workers}",
        flush=True,
    )

    config = {
        "domain": args.domain,
        "scaffold": args.scaffold,
        "metric": CEILING_METRIC,
        "seeds": args.seeds,
        "restarts": args.restarts,
        "n_samples": args.n_samples,
        "population_size": args.population_size,
        "generations": args.generations,
        "evaluations_per_fit": args.population_size * args.generations,
        "refit": args.refit_constants,
    }

    pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
    # `imap_unordered` rather than `map`, and a write after every problem. These
    # runs take hours, and a run killed at 90% that had buffered everything in
    # memory would leave nothing at all -- which has happened often enough here
    # to be worth designing against. Results arrive out of order, so the rows are
    # sorted on every write rather than once at the end.
    results = (
        pool.imap_unordered(probe, problem_ids) if pool is not None else map(probe, problem_ids)
    )

    rows: list[dict[str, Any]] = []
    for row in results:
        rows.append(row)
        write_artifacts(args.out, config, rows)
        print(
            f"[ceiling] {row['problem']:<32} {row['ratio']:>8.2f}x margin  "
            f"({len(rows)}/{len(problem_ids)})",
            flush=True,
        )

    if pool is not None:
        pool.close()
        pool.join()

    rows.sort(key=lambda row: row["ratio"], reverse=True)
    write_artifacts(args.out, config, rows)

    reachable = [row for row in rows if row["reachable"]]
    explanation = {
        "reachable": "can reach CONTRIBUTES",
        "faithful": "NULL: rule already picks the best",
        "tied": "NULL: restarts tie, nothing to pick",
    }
    print(f"\n{'problem':<32}{'x margin':>10}{'spread/marg':>13}  selection-only wrapper")
    for row in rows:
        print(
            f"{row['problem']:<32}{row['ratio']:>9.2f}x{row['spread_ratio']:>12.2f}x  "
            f"{explanation[row['null_because']]}"
        )

    tied = [row for row in rows if row["null_because"] == "tied"]
    faithful = [row for row in rows if row["null_because"] == "faithful"]
    print(
        f"\n[ceiling] {len(reachable)} of {len(rows)} problems leave any room at all "
        f"for a selection-only wrapper to be credited."
    )
    print(
        f"[ceiling] of the {len(rows) - len(reachable)} that do not: {len(faithful)} because "
        f"the pipeline's own rule already picks the best candidate, {len(tied)} because the "
        f"restarts tie and there is nothing to pick between."
    )
    print(f"[ceiling] wrote {args.out / 'ceiling.json'} and {args.out / 'ceiling.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
