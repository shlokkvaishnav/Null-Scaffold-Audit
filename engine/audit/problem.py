"""What an audit runs over, and who supplies it.

The audit compares a wrapper against its own base searcher. Both arms need the
same problem, and only a domain knows how to build one -- which problems exist,
how to sample them, what the right answer is.

So the problem is domain-supplied and the *shape* is agreed here. That is a
narrower claim than it looks: `run_arms` still takes `problem: Any` and never
reads a field of it. This dataclass exists so two plugins can hand the same
thing to the same scaffold without importing each other, not so the engine can
inspect it.

`ground_truth` is deliberately a plain dict rather than a typed record. What
counts as the true answer is the domain's business; the audit only forwards it
to whatever checks equivalence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["AuditProblem", "AuditProblemSource"]


@dataclass(frozen=True)
class AuditProblem:
    """One benchmark instance, already split into train and test."""

    equation_id: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    ground_truth: dict[str, Any]


@runtime_checkable
class AuditProblemSource(Protocol):
    """A domain's catalogue of auditable problems.

    Registering one of these is what makes a domain auditable. It is kept
    separate from `DomainPlugin` on purpose: a domain can be perfectly usable
    through the orchestrator without being set up for an audit, and requiring
    both would make the audit a tax on adding a domain rather than a tool for
    checking one.
    """

    name: str

    def list_problems(self) -> Sequence[str]:
        """Every problem id this domain can build, in a stable order."""
        ...

    def build_problem(self, problem_id: str, *, n_samples: int, seed: int) -> AuditProblem:
        """Construct one problem. Must be deterministic in `seed`.

        A source returning different data for the same seed makes every verdict
        unreproducible, and the audit has no way to detect that from outside.
        """
        ...
