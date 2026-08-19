"""Decide whether a two-arm audit can answer anything, before paying for one.

    python scripts/audit_feasibility.py --domain physics \
        --ceiling results/selection_ceiling results/selection_ceiling_rest \
        --null-calibration results/calibration/null

A budget-matched audit compares a wrapper against its own primitive. Two
quantities decide in advance whether that comparison can say anything at all,
and both are cheaper than the comparison itself:

* The **selection ceiling** -- the most a selection-only wrapper could possibly
  gain. Measured by `scripts/measure_selection_ceiling.py`, in one arm.
* The **minimum detectable effect** -- the smallest difference the design could
  certify at a given seed count. Measured here, from a scaffold that is null by
  construction, so it describes the design's noise floor rather than any real
  wrapper's behaviour.

If the ceiling sits below the minimum detectable effect the audit is *futile*:
the largest effect physically available is smaller than the smallest one the
design could see, so the sweep cannot produce an informative answer whichever
way the truth falls. Running it anyway costs hours and returns INCONCLUSIVE,
which reads as a failure of evidence rather than what it is -- a property of the
design, knowable in advance.

Neither idea is new. The minimum detectable effect is standard experimental
design and the ceiling is the oracle gap under another name. What they are not
is *used*, in this literature, before the compute is spent: this project ran
four two-arm sweeps that could not have resolved anything, and this script is
the check that would have said so first.

The measurement is only honest if the calibration really is null. A scaffold
that quietly differs from the control would inflate the spread and make
everything look futile, so `--null-calibration` should point at a run of
`null_calibration` and nothing else.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_null_scaffold_audit import default_registry

METRIC = "rmse"
SEED_COUNTS = (20, 40, 100, 500)


def minimum_detectable_effect(spread: float, n: int, alpha: float, power: float) -> float:
    """Smallest difference a paired design of ``n`` could certify at this spread.

    The textbook form, stated one-sided at ``alpha`` to match the audit's per-side
    rate. It scales as 1/sqrt(n), which is why a design short by an order of
    magnitude cannot be rescued by doubling the seeds.
    """
    return float((stats.norm.ppf(1.0 - alpha) + stats.norm.ppf(power)) * spread / np.sqrt(n))


def test_rmse(problem: Any, equation: str) -> float | None:
    """Re-evaluate a recorded symbolic candidate on the held-out split.

    Recomputed from the stored representation rather than read from the report,
    because reports keep aggregate verdicts and not per-seed values. Both arms go
    through this same path, so any evaluation quirk cancels in the difference.

    Symbolic-domain specific: assumes ``equation`` is an expression string, as
    the archived physics/synthetic plugins produced. Imported here rather than
    at module scope, matching the convention in measure_selection_ceiling.py --
    a domain that has no equation strings (e.g. NAS, where a candidate is an
    architecture encoding) has no use for this function and should not need
    engine.expressions importable to import this module at all.
    """
    try:
        from engine.expressions.hypothesis import Hypothesis

        predicted = np.asarray(
            Hypothesis(equation=equation, regime_id=0, iteration=0).evaluate(problem.x_test),
            dtype=float,
        )
    except Exception:  # noqa: BLE001
        return None
    fallback = float(np.mean(np.asarray(problem.y_train, dtype=float)))
    predicted = np.where(np.isfinite(predicted), predicted, fallback)
    truth = np.asarray(problem.y_test, dtype=float)
    return float(np.sqrt(np.mean((truth - predicted) ** 2)))


def load_ceilings(directories: list[Path]) -> dict[str, dict[str, Any]]:
    ceilings: dict[str, dict[str, Any]] = {}
    for directory in directories:
        path = directory / "ceiling.json"
        if not path.exists():
            print(f"[feasibility] no ceiling.json in {directory}", file=sys.stderr)
            continue
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            ceilings[row["problem"]] = row
    return ceilings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--ceiling", nargs="+", type=Path, required=True)
    parser.add_argument("--null-calibration", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "feasibility")
    args = parser.parse_args()

    ceilings = load_ceilings(list(args.ceiling))
    calibration = json.loads((args.null_calibration / "audit.json").read_text(encoding="utf-8"))
    source = default_registry().build_problem_source(args.domain)
    observed_n = int(calibration["config"]["seeds"])

    rows: list[dict[str, Any]] = []
    for report in calibration["reports"]:
        problem_id = report["equation_id"]
        if problem_id not in ceilings:
            print(f"[feasibility] no ceiling for {problem_id}, skipping", file=sys.stderr)
            continue
        problem = source.build_problem(problem_id, n_samples=args.n_samples, seed=0)

        treatment = [test_rmse(problem, e) for e in report["treatment_representations"]]
        control = [test_rmse(problem, e) for e in report["control_representations"]]
        paired = [
            (t, c)
            for t, c in zip(treatment, control, strict=True)
            if t is not None and c is not None
        ]
        if len(paired) < 2:
            print(f"[feasibility] {problem_id}: too few evaluable pairs", file=sys.stderr)
            continue

        differences = np.array([c - t for t, c in paired], dtype=float)
        spread = float(differences.std(ddof=1))
        ceiling = float(ceilings[problem_id]["ceiling"])
        mdes = {
            n: minimum_detectable_effect(spread, n, args.alpha, args.power) for n in SEED_COUNTS
        }
        here = minimum_detectable_effect(spread, observed_n, args.alpha, args.power)

        rows.append(
            {
                "domain": args.domain,
                "problem": problem_id,
                "metric": METRIC,
                "ceiling": ceiling,
                "null_spread": spread,
                "seeds_measured": observed_n,
                "mde_at_measured_n": here,
                **{f"mde_at_{n}": mdes[n] for n in SEED_COUNTS},
                # Above 1 the design can in principle resolve the largest effect
                # available. At or below it, no reachable seed count helps enough to
                # matter -- the ceiling is a bound on the effect, not a noisy estimate
                # of it, so there is nothing larger waiting to be resolved.
                "ceiling_over_mde": ceiling / here if here else float("inf"),
                "feasible": bool(ceiling > here),
                "seeds_to_feasible": next((n for n in SEED_COUNTS if ceiling > mdes[n]), None),
            }
        )

    if not rows:
        print("[feasibility] nothing to report", file=sys.stderr)
        return 1

    rows.sort(key=lambda row: row["ceiling_over_mde"], reverse=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "feasibility.json").write_text(
        json.dumps({"config": {k: str(v) for k, v in vars(args).items()}, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    with (args.out / "feasibility.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'problem':<24}{'ceiling':>11}{'MDE':>11}{'ratio':>9}  audit worth running?")
    for row in rows:
        if row["feasible"]:
            verdict = "yes"
        elif row["seeds_to_feasible"]:
            verdict = f"not at {observed_n} seeds; needs {row['seeds_to_feasible']}"
        else:
            verdict = "NO -- futile at every seed count checked"
        print(
            f"{row['problem']:<24}{row['ceiling']:>11.4g}{row['mde_at_measured_n']:>11.4g}"
            f"{row['ceiling_over_mde']:>8.2f}x  {verdict}"
        )
    feasible = sum(1 for row in rows if row["feasible"])
    print(
        f"\n[feasibility] {feasible} of {len(rows)} problems can resolve their own ceiling "
        f"at {observed_n} seeds."
    )
    print(f"[feasibility] wrote {args.out / 'feasibility.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
