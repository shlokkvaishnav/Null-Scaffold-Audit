"""Run the audit programme unattended, and refuse to run the parts that cannot answer.

    python scripts/research_queue.py --seed-queue --domain physics
    python scripts/research_queue.py --dry-run
    python scripts/research_queue.py --workers 4

The record this project has accumulated is eleven audits whose verdicts are almost
all ``INCONCLUSIVE``. That reads as a failure of evidence and is not one: at the seed
count they were run at, the design could not resolve the effect that was available.
`scripts/audit_feasibility.py` says so in advance, and says what seed count would
work. Applying that one problem at a time by hand is the only thing standing between
here and a results table, and it is mechanical -- every decision in it is numeric and
was fixed before any of it ran.

So this drives it. Each **cell** is one (problem, scaffold) pair, and moves through a
chain that gets more expensive at every step and can terminate at any of them:

    ceiling      -- already measured, one arm. Bounds selection-only scaffolds.
    gate         -- a null calibration per *problem*, shared by all its cells.
    feasibility  -- free; compares the ceiling against the detectable effect.
    audit        -- the two-arm sweep, run only at the seed count the gate prescribed.

WHAT THIS IS NOT ALLOWED TO DO
------------------------------
It generates evidence. It never modifies the instrument. Every serious error in this
project's history -- inverted p-values, a bound asserted from a point estimate, a guard
placed in the wrong function, a sweep whose arms were byte-identical on every seed --
was caught by reasoning about the statistics and by no amount of running. A loop that
could edit `engine/audit/` would have committed all four and built a confident table on
top of them. So:

* It writes under ``results/`` and nowhere else, checked rather than intended.
* It never invokes git and never edits source.
* It never re-runs a cell that reached a terminal state. A loop that retries until it
  likes the answer is a garden of forking paths with a machine tending it.
* ``n_star`` is written to the state file *before* the audit command is built, and read
  back out of the state to build it. The pre-registered margins already work this way;
  a seed count that moved after seeing an interval would be the same violation.

``ESCALATE`` is the state that makes the rest of it safe. If the gate said ``n_star``
was enough and the audit still returned ``INCONCLUSIVE``, then the design's model of
itself is wrong -- and the one response that must not be available is raising the seed
count and going again. It stops and says so.

This names no scientific domain. It takes ``--domain`` and a scaffold path and hands
both to the same loaders the audit runner uses, so driving a different domain's
programme is a matter of arguments.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Both, and both needed. The repo root is for `engine`/`plugins`; this file's own
# directory is for the two sibling scripts below, which is already on the path when
# this runs as a script and is not when it is imported as `scripts.research_queue`,
# as the tests import it.
for _path in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from audit_feasibility import load_ceilings
from run_null_scaffold_audit import load_scaffold

RESULTS = REPO_ROOT / "results"
QUEUE = RESULTS / "queue"
STATE = QUEUE / "state.json"

NULL_SCAFFOLD = "plugins.physics.audit_adapter:null_calibration"

MINIMUM_SEEDS = 40
"""Floor on any audit's seed count, and not a round number.

`audit_feasibility` sizes the design from the rmse spread alone, but the overall
verdict is an intersection-union test over three metrics, so the binding constraint
can be one it does not model -- and is. `exact_recovery` compares a paired difference
of proportions against a 0.10 margin, and at 20 seeds the tightest interval reachable
at all, with zero discordant pairs and perfect agreement, is +/-0.119. No data quality
certifies 0.10 from 20 seeds, which is why that metric vetoed every overall NULL this
project could produce. At 40 seeds the same interval is +/-0.063. Both measured.
"""

TERMINAL = {"CLOSED", "FUTILE", "NULL", "CONTRIBUTES", "HARMFUL", "ESCALATE", "FAILED"}
VERDICTS = {"NULL", "CONTRIBUTES", "HARMFUL"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _within_results(path: Path) -> Path:
    """Refuse any output path outside ``results/``.

    Checked on the resolved path rather than the string, so a path cannot walk out
    with ``..``. This allowlist is the whole safety story for leaving the queue
    running unattended, and a safety story that holds only while nobody passes the
    wrong flag is not one.
    """
    resolved = path.resolve()
    root = RESULTS.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"refusing to write outside results/: {resolved}")
    return resolved


def load_state() -> dict[str, Any]:
    if not STATE.exists():
        return {"gates": {}, "cells": [], "ceilings": {}}
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state.setdefault("gates", {})
    state.setdefault("ceilings", {})
    state.setdefault("cells", [])
    return state


def save_state(state: dict[str, Any]) -> None:
    _within_results(STATE)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _short(scaffold: str) -> str:
    return scaffold.rsplit(":", 1)[-1]


def run_stage(command: list[str], *, dry: bool, log: Path | None = None) -> tuple[int, float]:
    """Run one stage, returning its exit code and how long it took.

    The duration goes into the state file because the queue's cost model is currently
    in evaluation counts, which are not minutes. After one pass it is in minutes,
    measured on the machine that will run the rest.
    """
    if dry:
        print("  would run:", " ".join(command))
        return 0, 0.0
    started = time.monotonic()
    handle = log.open("w", encoding="utf-8") if log else None
    try:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    finally:
        if handle is not None:
            handle.close()
    return completed.returncode, time.monotonic() - started


def audit_command(
    args: argparse.Namespace, problem: str, scaffold: str, seeds: int, out: Path
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_null_scaffold_audit.py"),
        "--domain",
        args.domain,
        "--scaffold",
        scaffold,
        "--problems",
        problem,
        "--seeds",
        str(seeds),
        "--n-samples",
        str(args.n_samples),
        "--max-iters",
        str(args.max_iters),
        "--population-size",
        str(args.population_size),
        "--generations",
        str(args.generations),
        "--workers",
        str(args.workers),
        "--out",
        str(_within_results(out)),
    ]


def feasibility_command(args: argparse.Namespace, calibration: Path, out: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "audit_feasibility.py"),
        "--domain",
        args.domain,
        "--ceiling",
        *[str(directory) for directory in args.ceiling],
        "--null-calibration",
        str(calibration),
        "--n-samples",
        str(args.n_samples),
        "--out",
        str(_within_results(out)),
    ]


def read_verdict(out: Path, problem: str) -> str | None:
    path = out / "audit.json"
    if not path.exists():
        return None
    for report in json.loads(path.read_text(encoding="utf-8"))["reports"]:
        if report["equation_id"] == problem:
            return str(report["verdict"])
    return None


def resolve_gate(args: argparse.Namespace, state: dict[str, Any], problem: str) -> dict[str, Any]:
    """Run, or reuse, the null calibration that sizes every audit on this problem.

    Keyed by problem rather than by cell on purpose. The calibration measures the
    design's noise floor, which is a property of the problem and the budget and not of
    whichever scaffold is being audited against it -- so several scaffolds on one
    problem share one calibration instead of paying for identical copies of it. It is
    also not pure overhead: a scaffold that is null by construction and fails to read
    NULL is a finding about the instrument, which is why the verdict is kept next to
    the spread.
    """
    gate = state["gates"].setdefault(problem, {"stage": "PENDING"})
    if gate["stage"] in {"READY", "FUTILE", "FAILED", "DRY"}:
        return gate

    out = QUEUE / f"gate__{problem}"
    log = QUEUE / f"gate__{problem}.log"
    print(f"[queue] gate {problem}: null calibration at {MINIMUM_SEEDS} seeds", flush=True)
    code, seconds = run_stage(
        audit_command(args, problem, NULL_SCAFFOLD, MINIMUM_SEEDS, out),
        dry=args.dry_run,
        log=None if args.dry_run else log,
    )
    if args.dry_run:
        run_stage(feasibility_command(args, out, out), dry=True)
        # Marked even on a dry run. Several cells share one gate, and a dry run that
        # printed it once per cell would report three times the calibrations it is
        # actually going to pay for -- which defeats the only thing a dry run is for.
        gate["stage"] = "DRY"
        return gate

    gate["seconds"] = round(seconds, 1)
    gate["calibration_verdict"] = read_verdict(out, problem)
    if code != 0:
        gate.update(stage="FAILED", error=f"calibration exited {code}; see {log.name}")
        return gate

    code, _ = run_stage(
        feasibility_command(args, out, out),
        dry=False,
        log=QUEUE / f"feasibility__{problem}.log",
    )
    if code != 0:
        gate.update(stage="FAILED", error=f"feasibility exited {code}")
        return gate

    payload = json.loads((out / "feasibility.json").read_text(encoding="utf-8"))
    row = next((r for r in payload["rows"] if r["problem"] == problem), None)
    if row is None:
        gate.update(stage="FAILED", error="feasibility produced no row for this problem")
        return gate

    gate["null_spread"] = row["null_spread"]
    gate["ceiling"] = row["ceiling"]
    if row["seeds_to_feasible"] is None:
        # The bound sits below the noise floor at every seed count checked. That is not
        # a gap in the evidence, it is a property of the design, and it is the cheapest
        # true thing available here.
        gate.update(stage="FUTILE", n_star=None)
    else:
        gate.update(stage="READY", n_star=max(int(row["seeds_to_feasible"]), MINIMUM_SEEDS))
    return gate


def advance(args: argparse.Namespace, state: dict[str, Any], cell: dict[str, Any]) -> None:
    problem, scaffold = cell["problem"], cell["scaffold"]

    if cell.get("selection_only"):
        ceiling = state.get("ceilings", {}).get(problem)
        if (
            ceiling
            and ceiling.get("ceiling_upper") is not None
            and (ceiling["ceiling_upper"] < ceiling["margin"])
        ):
            # The most that better selection could ever be worth here is under the
            # margin, so this scaffold is null by construction and a sweep would spend
            # hours rediscovering it. Read off the upper bound and never the mean: the
            # claim is a bound, and this project has asserted one from a point estimate
            # before now.
            cell.update(
                stage="CLOSED",
                detail=(
                    f"ceiling_upper={ceiling['ceiling_upper']:.4g} < margin={ceiling['margin']:.4g}"
                ),
            )
            return

    gate = resolve_gate(args, state, problem)
    if args.dry_run:
        return
    if gate["stage"] == "FAILED":
        cell.update(stage="FAILED", detail=gate.get("error"))
        return
    if gate["stage"] == "FUTILE":
        cell.update(
            stage="FUTILE", detail="no reachable seed count resolves this problem's ceiling"
        )
        return

    # Written before the command is built and read back out of the state to build it.
    # The point is that no path exists where the number the audit runs at is chosen
    # after anything about that audit is known.
    cell["n_star"] = gate["n_star"]
    save_state(state)

    if scaffold == NULL_SCAFFOLD and cell["n_star"] == MINIMUM_SEEDS:
        # This cell and its own gate are the same run: same scaffold, same seed count,
        # same problem. Paying for it twice would buy a second sample of the same
        # quantity and report it as a second result, which is worse than merely
        # wasteful. The gate's verdict is this cell's verdict.
        cell.update(
            stage=gate["calibration_verdict"] or "FAILED",
            detail=f"adopted from the gate; the calibration at n={MINIMUM_SEEDS} is this cell",
        )
        return

    stem = f"{problem}__{_short(scaffold)}"
    out = QUEUE / stem
    log = QUEUE / f"{stem}.log"
    print(f"[queue] audit {problem} x {_short(scaffold)} at {cell['n_star']} seeds", flush=True)
    code, seconds = run_stage(
        audit_command(args, problem, scaffold, cell["n_star"], out), dry=False, log=log
    )
    cell["seconds"] = round(seconds, 1)
    if code != 0:
        cell.update(stage="FAILED", detail=f"audit exited {code}; see {log.name}")
        return

    verdict = read_verdict(out, problem)
    if verdict in VERDICTS:
        cell.update(stage=verdict, detail=None)
    elif verdict == "INCONCLUSIVE":
        # The gate said this seed count sufficed and it did not. Something in the
        # design's model of itself is wrong -- the spread estimate, a guard that fired,
        # an arm that did not vary. Raising the seeds and going again would convert a
        # real signal about the instrument into a result about the scaffold.
        cell.update(stage="ESCALATE", detail=f"INCONCLUSIVE at the prescribed n={cell['n_star']}")
    else:
        cell.update(stage="FAILED", detail=f"no verdict for {problem} in {out.name}")


def seed_queue(args: argparse.Namespace) -> dict[str, Any]:
    ceilings = load_ceilings(list(args.ceiling))
    state: dict[str, Any] = {"gates": {}, "cells": [], "ceilings": {}, "seeded": _now()}

    for problem in args.problems:
        row = ceilings.get(problem)
        if row is None:
            print(
                f"[queue] no ceiling for {problem}; run measure_selection_ceiling.py first",
                file=sys.stderr,
            )
            continue
        # `ceiling_upper` only if the artifact carries it. Every ceiling measured
        # before the bound was added reports a mean and nothing else, and comparing a
        # mean to a margin is exactly the error that bound exists to prevent -- so a
        # stale artifact disables the shortcut rather than being read as if it were
        # current. The cost is a sweep; the alternative is a skip nobody licensed.
        state["ceilings"][problem] = {
            "ceiling_upper": row.get("ceiling_upper"),
            "margin": row["margin"],
        }
        for scaffold in args.scaffolds:
            state["cells"].append(
                {
                    "domain": args.domain,
                    "problem": problem,
                    "scaffold": scaffold,
                    # Asked of the scaffold rather than listed here. A scaffold that
                    # does not declare it is treated as unbounded by the ceiling, so
                    # the omission costs a sweep instead of licensing a skip.
                    "selection_only": bool(
                        getattr(load_scaffold(scaffold), "selection_only", False)
                    ),
                    "stage": "QUEUED",
                    "n_star": None,
                    "seconds": None,
                    "detail": None,
                }
            )
    return state


def summarize(state: dict[str, Any]) -> None:
    print(f"\n{'problem':<22}{'scaffold':<24}{'n*':>5}  stage")
    for cell in state["cells"]:
        star = cell["n_star"] or "-"
        detail = f"   {cell['detail']}" if cell.get("detail") else ""
        print(
            f"{cell['problem']:<22}{_short(cell['scaffold']):<24}{star:>5}  {cell['stage']}{detail}"
        )

    tally: dict[str, int] = {}
    for cell in state["cells"]:
        tally[cell["stage"]] = tally.get(cell["stage"], 0) + 1
    print(f"\n[queue] {tally}")

    escalated = [c for c in state["cells"] if c["stage"] == "ESCALATE"]
    if escalated:
        print(
            f"[queue] {len(escalated)} cell(s) need a human: the gate sized them and they "
            f"still could not resolve, so the design's model of itself is wrong."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="physics")
    parser.add_argument(
        "--seed-queue", action="store_true", help="Enumerate cells into state.json and stop."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print every command that would run; run none."
    )
    parser.add_argument(
        "--problems", nargs="*", default=["boltzmann_factor", "coulomb_force", "rc_discharge"]
    )
    parser.add_argument(
        "--scaffolds",
        nargs="*",
        default=[
            "plugins.physics.audit_adapter:null_calibration",
            "plugins.physics.audit_adapter:wasteful_calibration",
            "plugins.physics.audit_adapter:oracle_calibration",
        ],
    )
    parser.add_argument(
        "--ceiling",
        nargs="+",
        type=Path,
        default=[RESULTS / "selection_ceiling", RESULTS / "selection_ceiling_rest"],
    )
    parser.add_argument("--only", default=None, help="Restrict this pass to one problem.")
    parser.add_argument(
        "--max-cells", type=int, default=None, help="Stop after this many cells advance."
    )
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    QUEUE.mkdir(parents=True, exist_ok=True)

    if args.seed_queue:
        state = seed_queue(args)
        save_state(state)
        print(f"[queue] seeded {len(state['cells'])} cells into {STATE}")
        summarize(state)
        return 0

    state = load_state()
    if not state["cells"]:
        print("[queue] nothing queued; run --seed-queue first", file=sys.stderr)
        return 2

    advanced = 0
    for cell in state["cells"]:
        if cell["stage"] in TERMINAL:
            continue
        if args.only and cell["problem"] != args.only:
            continue
        if args.max_cells is not None and advanced >= args.max_cells:
            break
        advance(args, state, cell)
        if not args.dry_run:
            save_state(state)
        advanced += 1

    if not args.dry_run:
        save_state(state)
    summarize(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
