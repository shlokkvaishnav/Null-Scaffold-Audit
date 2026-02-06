class AgentMemory:
    """
    Stores long-term regime transitions and valid hypotheses.
    Strategic curation, not archival dump.
    """
    def __init__(self, max_hypotheses_per_regime=5, max_capacity=1000):
        self.hypotheses = []
        self.transition_matrix = None
        self.regime_history = []
        self.max_hypotheses_per_regime = max_hypotheses_per_regime
        self.max_capacity = max_capacity
        self.pruned_log = []  # Track forgetting

    def store(self, hypothesis):
        """
        Store a hypothesis in memory.
        """
        self.hypotheses.append(hypothesis)

    def recall(self, regime_id=None, query=None):
        """
        Retrieve hypotheses matching query criteria.
        
        Args:
            regime_id: Optional regime ID to filter by
            query: Optional dict with filtering criteria
        """
        if regime_id is not None:
            return [h for h in self.hypotheses if h.regime_id == regime_id and h.valid]
        
        if query:
            return [h for h in self.hypotheses if h.valid]
        
        return self.hypotheses

    def prune(self, current_iteration=None):
        """
        Keep only top-K hypotheses per regime (strategic curation).
        
        Args:
            current_iteration: Current agent iteration for logging
        
        Returns:
            int: Number of hypotheses removed
        """
        initial_count = len(self.hypotheses)
        
        kept = []
        
        # Get unique regime IDs
        regime_ids = set(h.regime_id for h in self.hypotheses)
        
        for k in regime_ids:
            # Get all hypotheses for this regime
            candidates = [h for h in self.hypotheses if h.regime_id == k and h.valid]
            
            # Sort by score (descending)
            candidates.sort(key=lambda h: h.score if h.score is not None else float('-inf'), reverse=True)
            
            # Keep top K
            top_k = candidates[:self.max_hypotheses_per_regime]
            pruned = candidates[self.max_hypotheses_per_regime:]
            
            kept.extend(top_k)
            
            # Log pruned hypotheses (lineage tracking)
            for h in pruned:
                self.pruned_log.append({
                    "hypothesis": h,
                    "score": h.score,
                    "iteration": current_iteration
                })
        
        removed = initial_count - len(kept)
        self.hypotheses = kept
        
        return removed
    
    def export_rejections(self):
        """
        Export rejection statistics for explainability and analysis.
        
        Returns:
            list: Rejection records with regime and violation info
        """
        if not hasattr(self, 'rejection_log'):
            return []
        
        return [
            {
                "regime": entry["hypothesis"].regime_id,
                "equation": entry["hypothesis"].equation,
                "violations": entry["violations"]
            }
            for entry in self.rejection_log
        ]
