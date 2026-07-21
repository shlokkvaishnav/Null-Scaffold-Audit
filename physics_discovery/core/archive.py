"""
Hypothesis Archive Module.
Strategic curation of hypotheses with selective forgetting.
"""

from typing import List, Dict, Optional, Any


class HypothesisArchive:
    """
    Stores long-term regime transitions and valid hypotheses.
    Strategic curation, not archival dump.
    """

    def __init__(self, max_hypotheses_per_regime: int = 5, max_capacity: int = 1000):
        """
        Initialize the hypothesis archive.

        Args:
            max_hypotheses_per_regime: Maximum hypotheses to keep per regime
            max_capacity: Overall capacity limit
        """
        self.hypotheses: List = []
        self.transition_matrix = None
        self.regime_history: List = []
        self.max_hypotheses_per_regime = max_hypotheses_per_regime
        self.max_capacity = max_capacity
        self.rejection_log: List[Dict] = []
        self.pruned_log: List[Dict] = []

    def store(self, hypothesis) -> None:
        """
        Store a hypothesis in the archive.

        Args:
            hypothesis: Hypothesis object to store
        """
        if hypothesis is not None:
            self.hypotheses.append(hypothesis)

    def recall(self, regime_id: Optional[int] = None, query: Optional[Dict] = None) -> List:
        """
        Retrieve hypotheses matching query criteria.

        Args:
            regime_id: Optional regime ID to filter by
            query: Optional dict with filtering criteria

        Returns:
            list: Matching hypotheses
        """
        if not self.hypotheses:
            return []

        if regime_id is not None:
            return [h for h in self.hypotheses
                    if h.regime_id == regime_id and getattr(h, 'valid', True)]

        if query:
            return [h for h in self.hypotheses
                    if getattr(h, 'valid', True)]

        return self.hypotheses.copy()

    def prune(self, current_iteration: Optional[int] = None) -> int:
        """
        Keep only top-K hypotheses per regime (strategic curation).

        Args:
            current_iteration: Current agent iteration for logging

        Returns:
            int: Number of hypotheses removed
        """
        if not self.hypotheses:
            return 0

        initial_count = len(self.hypotheses)
        kept = []

        # Get unique regime IDs
        regime_ids = set(h.regime_id for h in self.hypotheses if hasattr(h, 'regime_id'))

        for k in regime_ids:
            # Get valid hypotheses for this regime
            candidates = [h for h in self.hypotheses
                         if getattr(h, 'regime_id', None) == k
                         and getattr(h, 'valid', True)]

            # Sort by score (descending), handle None scores
            candidates.sort(
                key=lambda h: getattr(h, 'score', None) or float('-inf'),
                reverse=True
            )

            # Keep top K
            top_k = candidates[:self.max_hypotheses_per_regime]
            pruned = candidates[self.max_hypotheses_per_regime:]

            kept.extend(top_k)

            # Log pruned hypotheses
            for h in pruned:
                self.pruned_log.append({
                    "hypothesis": h,
                    "score": getattr(h, 'score', None),
                    "iteration": current_iteration
                })

        # Also keep invalid hypotheses (for analysis)
        invalid = [h for h in self.hypotheses
                   if not getattr(h, 'valid', True)]

        self.hypotheses = kept
        return initial_count - len(kept)

    def log_rejection(self, hypothesis) -> None:
        """
        Log a rejected hypothesis for analysis.

        Args:
            hypothesis: Rejected hypothesis
        """
        self.rejection_log.append({
            "hypothesis": hypothesis,
            "violations": getattr(hypothesis, 'violation_log', {})
        })

    def get_rejection_count(self) -> int:
        """Get total number of rejections."""
        return len(self.rejection_log)

    def get_pruned_count(self) -> int:
        """Get total number of pruned hypotheses."""
        return len(self.pruned_log)

    def export_rejections(self) -> List[Dict]:
        """
        Export rejection statistics for explainability and analysis.

        Returns:
            list: Rejection records with regime and violation info
        """
        return [
            {
                "regime": getattr(entry["hypothesis"], 'regime_id', None),
                "equation": getattr(entry["hypothesis"], 'equation', None),
                "violations": entry.get("violations", {})
            }
            for entry in self.rejection_log
        ]

    def clear(self) -> None:
        """Clear the archive."""
        self.hypotheses = []
        self.rejection_log = []
        self.pruned_log = []
        self.regime_history = []
