"""Run the null-scaffold audit against a registered pipeline.

    python scripts/run_null_scaffold_audit.py \
        --domain physics \
        --scaffold plugins.physics.audit_adapter:DiscoveryAgentScaffold \
        --subset smoke --seeds 10 --workers 8

Writes JSON and CSV artifacts under `results/null_scaffold_audit/`. Every number
this project reports about the audit comes from these files; none
is typed into prose by hand.

This script names no scientific domain. It asks the plugin registry for a
problem source by name and imports the scaffold from a path given on the command
line, so auditing a new domain means registering a source in that domain's
plugin and changing nothing here. Both are required rather than defaulted: a
default naming one domain would contradict the sentence above, and it also makes
every recorded command self-describing. That is the property `--domain synthetic`
exists to demonstrate: if the audit only ever ran against the domain it was
written for, "domain independent" would be an intention rather than a result.

PRE-REGISTERED MARGINS
----------------------
The practical-equivalence margins below were fixed before the first sweep was
run and are not to be adjusted after seeing an interval. An audit whose margin
moves to fit its results is not an audit.

- `rmse`:                 5% of the test target's standard deviation.
- `symbolic_complexity`:  2 nodes.
- `exact_recovery`:       0.10 (10 percentage points).

The rmse margin is scaled by the problem rather than fixed absolutely because
benchmark targets span many orders of magnitude, so a single absolute number
would be vacuous on one problem and unattainable on another. It is scaled by the
*target*, not by either arm's error, so it remains a property of the problem and
cannot be influenced by the results.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.audit import AuditProblem, NotSeparableError, Scaffold, Verdict, audit
from engine.discovery import discover_plugins
from engine.registry import PluginRegistry

RMSE_MARGIN_FRACTION = 0.05
COMPLEXITY_MARGIN = 2.0
RECOVERY_MARGIN = 0.10

HIGHER_IS_BETTER = {
    "rmse": False,
    "symbolic_complexity": False,
    "exact_recovery": True,
}

SMOKE_SUBSET_SIZE = 8


def default_registry() -> PluginRegistry:
    """Every plugin advertised through the `sde.plugins` entry-point group."""
    registry = PluginRegistry()
    discover_plugins(registry)
    return registry


def load_scaffold(path: str, **kwargs: Any) -> Scaffold:
    """Import and construct a scaffold from a ``module:attribute`` path.

    Resolved at run time rather than imported at module scope so this script
    stays importable -- and testable -- without any particular plugin's
    dependencies installed.
    """
    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"--scaffold must look like 'module:Attribute', got {path!r}")
    module = importlib.import_module(module_name)
    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"{module_name} has no attribute {attribute!r}") from exc
    return factory(**kwargs)


def margins_for(problem: AuditProblem) -> dict[str, float]:
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
    parser.add_argument(
        "--domain",
        default=None,
        help="Registered audit problem source (see --list-domains). Required.",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="Print the registered problem sources and exit.",
    )
    parser.add_argument("--scaffold", default=None, help="module:Attribute to audit. Required.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes to spread seeds across. Seeds are independent, so this "
        "changes wall-clock only -- never a verdict.",
    )
    parser.add_argument(
        "--common-random-numbers",
        action="store_true",
        help="Pair the arms on identical random streams. Reduces the variance of "
        "the paired difference, and narrows the question: both arms then run the "
        "same searches, so the comparison isolates what the wrapper does with "
        "them rather than which regions it happens to explore.",
    )
    parser.add_argument("--subset", choices=["smoke", "all"], default="smoke")
    parser.add_argument("--problems", nargs="*", default=None, help="Explicit problem ids.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "null_scaffold_audit")
    args = parser.parse_args()

    registry = default_registry()

    if args.list_domains:
        print(f"[audit] registered problem sources: {registry.list_problem_sources()}")
        return 0

    if not args.domain or not args.scaffold:
        parser.error(
            "--domain and --scaffold are both required. "
            f"Registered domains: {registry.list_problem_sources()}"
        )

    try:
        source = registry.build_problem_source(args.domain)
    except KeyError as exc:
        print(f"[audit] {exc}", file=sys.stderr)
        return 2

    available = list(source.list_problems())
    if args.problems:
        unknown = [p for p in args.problems if p not in available]
        if unknown:
            print(f"[audit] {args.domain} has no problem(s) {unknown}", file=sys.stderr)
            return 2
        problem_ids = list(args.problems)
    elif args.subset == "smoke":
        problem_ids = available[:SMOKE_SUBSET_SIZE]
    else:
        problem_ids = available

    seeds = list(range(args.seeds))
    scaffold = load_scaffold(
        args.scaffold,
        max_iters=args.max_iters,
        population_size=args.population_size,
        generations=args.generations,
    )

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    config: dict[str, Any] = {
        "domain": args.domain,
        "scaffold": args.scaffold,
        "seeds": len(seeds),
        "max_iters": args.max_iters,
        "population_size": args.population_size,
        "generations": args.generations,
        "n_samples": args.n_samples,
        "workers": args.workers,
        "common_random_numbers": args.common_random_numbers,
        "evaluations_per_fit": args.population_size * args.generations,
        "margins": {
            "rmse": f"{RMSE_MARGIN_FRACTION} * std(y_test)",
            "symbolic_complexity": COMPLEXITY_MARGIN,
            "exact_recovery": RECOVERY_MARGIN,
        },
    }

    # Seeds are independent by construction, and both Pool.map and the builtin
    # preserve input order, so this changes wall-clock and nothing else. It has
    # to be processes rather than threads: the adapter counts real fits by
    # patching a class attribute, which is process-wide state.
    pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
    map_fn = pool.map if pool is not None else map

    print(
        f"[audit] domain={args.domain} scaffold={args.scaffold} "
        f"problems={len(problem_ids)} workers={args.workers}"
    )

    for problem_id in problem_ids:
        problem = source.build_problem(problem_id, n_samples=args.n_samples, seed=0)
        print(f"[audit] {problem_id}: {len(seeds)} seeds", flush=True)
        try:
            report = audit(
                scaffold,
                problem,
                seeds,
                margins=margins_for(problem),
                higher_is_better=HIGHER_IS_BETTER,
                map_fn=map_fn,
                common_random_numbers=args.common_random_numbers,
            )
        except NotSeparableError as exc:
            print(f"[audit] {problem_id}: NOT_SEPARABLE ({exc})", flush=True)
            rows.append(
                {"equation_id": problem_id, "metric": "-", "verdict": Verdict.NOT_SEPARABLE.value}
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
                    "domain": args.domain,
                    "equation_id": problem_id,
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
                "domain": args.domain,
                "equation_id": problem_id,
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

    if pool is not None:
        pool.close()
        pool.join()

    verdict_counts: dict[str, int] = {}
    for report_row in reports:
        verdict_counts[report_row["verdict"]] = verdict_counts.get(report_row["verdict"], 0) + 1
    print(f"\n[audit] overall verdicts across {len(reports)} problems: {verdict_counts}")
    print(f"[audit] wrote {args.out / 'audit.json'} and {args.out / 'audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
