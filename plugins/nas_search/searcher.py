"""RandomSearch: this plugin's base searcher, per SPEC.md.

Samples uniformly, without replacement, from NATS-Bench's tabular topology
index and keeps the candidate with the best validation accuracy -- the
control arm's ``B_restart`` primitive (``AUDIT_METHODOLOGY.md`` §4.1) for
this domain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.audit.arms import SearchOutcome

METRIC = "valid_accuracy"


@dataclass
class RandomSearch:
    """Samples ``budget`` architectures uniformly without replacement, keeps the best.

    ``restart_cost`` is ``budget``: one ``search()`` call spends exactly this
    many tabular lookups, which is what lets ``engine.audit.arms.run_arms``
    give the control arm the same total compute the treatment spent.

    Selection uses the same statistic ``search()`` reports (validation
    accuracy) -- not a held-out or otherwise privileged measure -- so the
    selection rule and the audited metric agree by construction and this
    searcher gives ``engine.audit.calibration.OracleScaffold`` nothing to
    exploit (``calibration.py``'s own note on why that scaffold's headroom
    must be checked, not assumed).
    """

    budget: int = 50

    @property
    def restart_cost(self) -> int:
        return self.budget

    def search(self, problem: Any, seed: int) -> SearchOutcome:
        # `problem: Any`, matching `engine.audit.arms.BaseSearcher.search`'s own
        # signature: this only needs `problem.num_archs` and
        # `problem.valid_accuracy(index)`, which any duck-typed stand-in can
        # provide (see the fakes in tests/test_plugin_nas_search.py) -- it does
        # not need to *be* a `NatsBenchProblem`.
        rng = np.random.default_rng(seed)
        size = min(self.budget, problem.num_archs)
        indices = rng.choice(problem.num_archs, size=size, replace=False)
        accuracies = [problem.valid_accuracy(int(idx)) for idx in indices]
        best = int(np.argmax(accuracies))
        # `intermediate_representations` is left at its default (empty): this
        # searcher does not expose the candidates it rejected within a single
        # call, only the winner. Degeneracy is still assessed correctly --
        # `IdentityRestartScaffold.run()` (via `NullScaffold`/`_RestartScaffold`)
        # collects one `representation` per restart (per `search()` call) into
        # *its own* `intermediate_representations`, which is what
        # `engine.audit.degeneracy.assess_degeneracy` reads. That is exactly
        # the check this experiment needs: whether independently seeded
        # restarts actually land on different architectures.
        return SearchOutcome(
            metrics={METRIC: float(accuracies[best])},
            evaluations_used=size,
            representation=str(int(indices[best])),
        )

    def select(self, outcomes: Sequence[SearchOutcome]) -> SearchOutcome:
        return max(outcomes, key=lambda outcome: outcome.metrics[METRIC])
