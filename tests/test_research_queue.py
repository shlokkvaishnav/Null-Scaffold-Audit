"""What the queue must never do when nobody is watching it.

These are not tests of the audit -- that is tested elsewhere and the queue does not
touch it. They pin the properties that make it safe to leave the queue running: it
cannot spend twice on the same question, it cannot choose a seed count after seeing a
result, it cannot quietly retry its way to an answer, and it cannot write outside
``results/``.

The subprocess layer is stubbed throughout, so the whole file runs in milliseconds and
never starts a sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import scripts.research_queue as queue


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every path the module writes to at a temporary directory."""
    results = tmp_path / "results"
    (results / "queue").mkdir(parents=True)
    monkeypatch.setattr(queue, "RESULTS", results)
    monkeypatch.setattr(queue, "QUEUE", results / "queue")
    monkeypatch.setattr(queue, "STATE", results / "queue" / "state.json")
    return results


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "domain": "physics",
        "dry_run": False,
        "n_samples": 500,
        "max_iters": 3,
        "population_size": 500,
        "generations": 20,
        "workers": 1,
        "ceiling": [Path("ceiling")],
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _cell(scaffold: str = "pkg:SomeScaffold", **overrides: object) -> dict[str, object]:
    cell: dict[str, object] = {
        "domain": "physics",
        "problem": "boltzmann_factor",
        "scaffold": scaffold,
        "selection_only": False,
        "stage": "QUEUED",
        "n_star": None,
        "seconds": None,
        "detail": None,
    }
    cell.update(overrides)
    return cell


def _ready_gate(n_star: int = 40) -> dict[str, object]:
    return {"stage": "READY", "n_star": n_star, "calibration_verdict": "NULL"}


def _stub_audit(
    monkeypatch: pytest.MonkeyPatch, verdict: str | None, *, code: int = 0
) -> list[list[str]]:
    """Record every command issued, and write the verdict the audit would have."""
    issued: list[list[str]] = []

    def fake_run(command: list[str], *, dry: bool, log: Path | None = None) -> tuple[int, float]:
        issued.append(command)
        if verdict is not None and "--out" in command:
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "audit.json").write_text(
                json.dumps({"reports": [{"equation_id": "boltzmann_factor", "verdict": verdict}]}),
                encoding="utf-8",
            )
        return code, 1.0

    monkeypatch.setattr(queue, "run_stage", fake_run)
    return issued


def test_terminal_cells_are_never_run_again(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The single property that separates a queue from a p-hacking machine.

    A loop free to re-run a cell it already resolved will, over enough passes, report
    whichever verdict it liked best. Every terminal state is checked, not just the
    convenient ones -- ``ESCALATE`` and ``FAILED`` most of all, because those are the
    ones a well-meaning retry would target.
    """
    issued = _stub_audit(monkeypatch, "NULL")
    for stage in sorted(queue.TERMINAL):
        state = {"gates": {}, "ceilings": {}, "cells": [_cell(stage=stage)]}
        queue.save_state(state)
        for _ in range(3):
            reloaded = queue.load_state()
            for cell in reloaded["cells"]:
                if cell["stage"] not in queue.TERMINAL:
                    queue.advance(_args(), reloaded, cell)
        assert issued == [], f"{stage} was re-run"


def test_audit_runs_at_the_seed_count_recorded_in_state(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``n_star`` reaches the command from the state file, not from a later decision.

    The margins are pre-registered and the seed count is now too. This checks the
    mechanism rather than the intent: whatever ``--seeds`` the audit receives must
    equal what the gate wrote down before the audit existed.
    """
    issued = _stub_audit(monkeypatch, "NULL")
    state = {
        "gates": {"boltzmann_factor": _ready_gate(100)},
        "ceilings": {},
        "cells": [_cell()],
    }
    queue.advance(_args(), state, state["cells"][0])

    assert state["cells"][0]["n_star"] == 100
    assert queue.load_state()["cells"][0]["n_star"] == 100, "n_star must be durable first"
    command = issued[-1]
    assert command[command.index("--seeds") + 1] == "100"


def test_inconclusive_at_the_prescribed_n_escalates_rather_than_retrying(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate said this would resolve and it did not, so the design is wrong.

    Exactly one audit may be issued. A second one would convert a real signal about the
    instrument into a result about the scaffold, which is the failure this whole
    subsystem exists to refuse.
    """
    issued = _stub_audit(monkeypatch, "INCONCLUSIVE")
    state = {"gates": {"boltzmann_factor": _ready_gate()}, "ceilings": {}, "cells": [_cell()]}
    queue.advance(_args(), state, state["cells"][0])

    assert state["cells"][0]["stage"] == "ESCALATE"
    assert len(issued) == 1
    assert "40" in str(state["cells"][0]["detail"])


def test_a_failing_stage_marks_the_cell_and_leaves_the_rest_alone(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One broken cell must not take the overnight run with it."""
    _stub_audit(monkeypatch, None, code=1)
    state = {"gates": {"boltzmann_factor": _ready_gate()}, "ceilings": {}, "cells": [_cell()]}
    queue.advance(_args(), state, state["cells"][0])

    assert state["cells"][0]["stage"] == "FAILED"
    assert "exited 1" in str(state["cells"][0]["detail"])


def test_the_null_cell_adopts_its_gate_instead_of_repeating_it(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate and the null cell are the same run; paying twice reports it twice."""
    issued = _stub_audit(monkeypatch, "NULL")
    state = {
        "gates": {"boltzmann_factor": _ready_gate(queue.MINIMUM_SEEDS)},
        "ceilings": {},
        "cells": [_cell(queue.NULL_SCAFFOLD)],
    }
    queue.advance(_args(), state, state["cells"][0])

    assert state["cells"][0]["stage"] == "NULL"
    assert issued == [], "the gate's calibration already answered this cell"


def test_a_closed_ceiling_settles_a_selection_only_scaffold_without_a_sweep(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issued = _stub_audit(monkeypatch, "NULL")
    state = {
        "gates": {},
        "ceilings": {"boltzmann_factor": {"ceiling_upper": 0.001, "margin": 0.05}},
        "cells": [_cell(selection_only=True)],
    }
    queue.advance(_args(), state, state["cells"][0])

    assert state["cells"][0]["stage"] == "CLOSED"
    assert issued == []


def test_a_ceiling_without_an_upper_bound_does_not_close_anything(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stale artifacts must cost a sweep, never license a skip.

    Every ceiling measured before the upper bound was added carries a mean and nothing
    else. Comparing that mean to a margin is precisely the error the bound was added to
    stop, so a row missing it has to disable the shortcut rather than fall back to the
    mean.
    """
    issued = _stub_audit(monkeypatch, "NULL")
    state = {
        "gates": {"boltzmann_factor": _ready_gate()},
        "ceilings": {"boltzmann_factor": {"ceiling_upper": None, "margin": 0.05}},
        "cells": [_cell(selection_only=True)],
    }
    queue.advance(_args(), state, state["cells"][0])

    assert state["cells"][0]["stage"] == "NULL"
    assert len(issued) == 1, "a stale ceiling must not skip the audit"


def test_a_futile_gate_terminates_its_cells_without_auditing_them(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issued = _stub_audit(monkeypatch, "NULL")
    state = {
        "gates": {"boltzmann_factor": {"stage": "FUTILE", "n_star": None}},
        "ceilings": {},
        "cells": [_cell()],
    }
    queue.advance(_args(), state, state["cells"][0])

    assert state["cells"][0]["stage"] == "FUTILE"
    assert issued == []


@pytest.mark.parametrize("escape", ["../outside", "../../etc", "../../../elsewhere"])
def test_writes_outside_results_are_refused(sandbox: Path, escape: str) -> None:
    """Checked on the resolved path, so ``..`` cannot walk out of the allowlist."""
    with pytest.raises(ValueError, match="outside results/"):
        queue._within_results(queue.RESULTS / escape)


def test_results_itself_and_its_children_are_allowed(sandbox: Path) -> None:
    assert queue._within_results(queue.RESULTS) == queue.RESULTS.resolve()
    assert queue._within_results(queue.QUEUE / "cell") == (queue.QUEUE / "cell").resolve()


def _stub_ceiling(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]], *, code: int = 0
) -> list[list[str]]:
    """Record every command issued, and write the ceiling table it would have produced."""
    issued: list[list[str]] = []

    def fake_run(command: list[str], *, dry: bool, log: Path | None = None) -> tuple[int, float]:
        issued.append(command)
        if code == 0 and "--out" in command:
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "ceiling.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
        return code, 1.0

    monkeypatch.setattr(queue, "run_stage", fake_run)
    return issued


def test_a_problem_with_a_bound_already_is_not_remeasured(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issued = _stub_ceiling(monkeypatch, rows=[])
    state = {
        "gates": {},
        "ceilings": {"boltzmann_factor": {"ceiling_upper": 0.001, "margin": 0.05}},
        "cells": [_cell(selection_only=True)],
    }
    queue.resolve_ceilings(_args(ceiling_seeds=10), state)

    assert issued == []


def test_ceiling_is_measured_once_for_every_pending_problem_not_once_each(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fan-out is the reason this stage exists; a per-problem loop would undo it."""
    issued = _stub_ceiling(
        monkeypatch,
        rows=[
            {
                "problem": "boltzmann_factor",
                "ceiling_upper": 0.001,
                "margin": 0.05,
                "status": "closed",
            },
            {"problem": "coulomb_force", "ceiling_upper": 0.02, "margin": 0.05, "status": "closed"},
        ],
    )
    state = {
        "gates": {},
        "ceilings": {},
        "cells": [
            _cell(selection_only=True),
            _cell(selection_only=True, problem="coulomb_force"),
        ],
    }
    queue.resolve_ceilings(_args(ceiling_seeds=10), state)

    assert len(issued) == 1, "one subprocess call must cover every pending problem"
    command = issued[0]
    assert "boltzmann_factor" in command
    assert "coulomb_force" in command
    assert state["ceilings"]["boltzmann_factor"]["ceiling_upper"] == 0.001
    assert state["ceilings"]["coulomb_force"]["ceiling_upper"] == 0.02


def test_a_failed_ceiling_measurement_leaves_cells_queued_with_the_shortcut_disabled(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing bound must cost a sweep, never a verdict."""
    issued = _stub_ceiling(monkeypatch, rows=[], code=1)
    state = {"gates": {}, "ceilings": {}, "cells": [_cell(selection_only=True)]}
    queue.resolve_ceilings(_args(ceiling_seeds=10), state)

    assert len(issued) == 1
    assert state["cells"][0]["stage"] == "QUEUED"
    assert "boltzmann_factor" not in state["ceilings"]
