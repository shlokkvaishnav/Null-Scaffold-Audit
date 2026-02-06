class AgentMemory:
    """
    Stores long-term regime transitions and valid hypotheses.
    """
    def __init__(self, max_capacity=1000):
        self.hypotheses = []
        self.transition_matrix = None
        self.regime_history = []
        self.max_capacity = max_capacity

    def store(self, hypothesis):
        """
        Store a hypothesis in memory.
        """
        self.hypotheses.append(hypothesis)

    def recall(self, query):
        """
        Retrieve hypotheses matching query criteria.
        """
        # TODO: Implement query logic
        return [h for h in self.hypotheses if h.valid]

    def prune(self, max_size=None):
        """
        Remove invalid hypotheses from memory.
        """
        max_size = max_size or self.max_capacity
        self.hypotheses = [h for h in self.hypotheses if h.valid]
        
        if len(self.hypotheses) > max_size:
            self.hypotheses = self.hypotheses[-max_size:]
        
        return len(self.hypotheses)
    
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
