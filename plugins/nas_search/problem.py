"""NATS-Bench topology search space as this plugin's audit problem.

Opaque to ``engine/audit`` per ``AUDIT_METHODOLOGY.md`` §4.4: nothing in
``engine/`` ever reads a field of :class:`NatsBenchProblem`, so nothing about
"architecture" leaks past this module. ``api`` is a
``nats_bench.api_topology.NATStopology`` handle, opaque in exactly the same
sense -- a candidate is an opaque integer index into it.

This deliberately does not reuse ``engine.audit.problem.AuditProblem``: that
dataclass's shape (``x_train``/``y_train``/``x_test``/``y_test``, a plain
regression split) is the physics/symbolic-regression plugin's, and a tabular
NAS benchmark has no such split -- there is no fitting to do, only lookups.
Forcing empty arrays into those fields to satisfy the shape would document
nothing and explain nothing. ``AuditProblemSource`` is a ``Protocol`` (duck
typed, not nominally enforced), so a domain-specific container that
implements its two methods is a conforming problem source without adopting a
shape built for a different domain.
"""

from __future__ import annotations

from typing import Any

CIFAR10_VALID = "cifar10-valid"
FULL_TRAINING_HP = "200"


class NatsBenchProblem:
    """One NATS-Bench topology-search-space instance, already loaded."""

    def __init__(self, *, dataset: str, hp: str, num_archs: int, api: Any) -> None:
        self.dataset = dataset
        self.hp = hp
        self.num_archs = num_archs
        self.api = api

    def valid_accuracy(self, index: int) -> float:
        """Validation accuracy for architecture ``index``.

        Fixed to ``is_random=False`` -- NATS-Bench's mean over its recorded
        training seeds for this architecture, rather than one of them drawn
        at random -- and called identically by every arm, so neither arm can
        manufacture a difference by reading a different statistic for the
        same architecture (SPEC.md, "Confounds considered": "Which statistic
        is read ... must be fixed in the AuditProblem implementation and held
        identical for both arms").
        """
        info = self.api.get_more_info(index, self.dataset, hp=self.hp, is_random=False)
        return float(info["valid-accuracy"])


class NatsBenchTopologyProblemSource:
    """This plugin's catalogue of auditable problems: one, the CIFAR-10 topology space."""

    name = "nas_search"

    def __init__(
        self, file_path: str, dataset: str = CIFAR10_VALID, hp: str = FULL_TRAINING_HP
    ) -> None:
        self._file_path = file_path
        self._dataset = dataset
        self._hp = hp
        self._api: Any = None

    def _get_api(self) -> Any:
        # Imported here, not at module scope: nats-bench is an optional
        # dependency (`pyproject.toml`'s `nas` extra), and importing it only
        # when a problem is actually built keeps this module importable --
        # and this plugin's other pieces usable -- without it installed.
        if self._api is None:
            import nats_bench

            self._api = nats_bench.create(self._file_path, "tss", fast_mode=True, verbose=False)
        return self._api

    def list_problems(self) -> list[str]:
        return ["cifar10-topology"]

    # Declared -> Any, not -> NatsBenchProblem: `AuditProblemSource.build_problem`
    # is typed as returning `engine.audit.problem.AuditProblem`, which (per this
    # module's docstring) `NatsBenchProblem` deliberately does not subclass.
    # `engine/` never checks the type at runtime -- `Protocol` is duck typed --
    # but mypy's structural check does, and `Any` says truthfully that this
    # source returns its own domain-specific shape rather than pretending to
    # return the physics plugin's.
    def build_problem(self, problem_id: str, *, n_samples: int, seed: int) -> Any:
        del n_samples, seed  # tabular lookup: nothing to sample or split at build time
        if problem_id not in self.list_problems():
            raise KeyError(f"unknown problem id {problem_id!r}; available: {self.list_problems()}")
        api = self._get_api()
        return NatsBenchProblem(dataset=self._dataset, hp=self._hp, num_archs=len(api), api=api)
