class SDMoSEAgent:
    """
    GRAIL-V Agent Orchestrator.
    Manages the Perception -> Retrieval -> Reasoning -> Verification -> Learning loop.
    """
    def __init__(self, config):
        self.config = config
        
        # Core components (to be initialized)
        self.perception = None
        self.experts = {}
        self.memory = None
        self.belief = None
        self.verifier = None
        
        # State
        self.current_hypotheses = []
        self.iteration = 0

    def observe(self, data):
        """
        Perception: Encode raw data into agent observations.
        Maps to: data/preprocess.py via PerceptionModule
        """
        if self.perception is None:
            from .perception import PerceptionModule
            self.perception = PerceptionModule(self.config)
        
        self.observation = self.perception.encode(data)
        return self.observation

    def retrieve(self):
        """
        Retrieval: Fetch relevant priors and historical regime patterns.
        Maps to: foundation init + priors (NOT learning)
        """
        if not hasattr(self, 'retrieval_module') or self.retrieval_module is None:
            from .retrieval import RetrievalModule
            self.retrieval_module = RetrievalModule(self.config)
        
        # Retrieve scientific priors
        self.priors = self.retrieval_module.retrieve_priors()
        
        # Retrieve candidate hypotheses from memory
        self.candidate_hypotheses = self.retrieval_module.retrieve_hypotheses(self.memory)
        
        return self.priors, self.candidate_hypotheses

    def reason(self):
        """
        Reasoning: Propose symbolic hypotheses for current regime.
        Maps to: PySR symbolic regression via ReasoningModule
        """
        if not hasattr(self, 'reasoning_module') or self.reasoning_module is None:
            from .reasoning import ReasoningModule
            self.reasoning_module = ReasoningModule(self.config)
        
        from .hypothesis import Hypothesis
        
        # Get number of regimes
        num_regimes = self.config.get("agent", {}).get("num_regimes", 3)
        
        self.proposed_hypotheses = []
        
        for k in range(num_regimes):
            equation = self.reasoning_module.propose_hypothesis(
                observation=self.observation,
                regime_id=k,
                priors=self.priors
            )
            
            h = Hypothesis(equation=equation, regime_id=k)
            self.proposed_hypotheses.append(h)
        
        return self.proposed_hypotheses

    def verify(self):
        """
        Verification: Validate hypotheses against physics constraints.
        Maps to: physics constraints (self-critique)
        """
        if not hasattr(self, 'verification_module') or self.verification_module is None:
            from .verification import VerificationModule
            self.verification_module = VerificationModule(self.config)
        
        self.verified_hypotheses = []
        
        for h in self.proposed_hypotheses:
            # Verify hypothesis
            h.verify(self.verification_module)
            
            # Score hypothesis (NEW: explicit evaluation)
            self.verification_module.score_hypothesis(h, self.observation)
            
            if h.valid:
                self.verified_hypotheses.append(h)
            else:
                # Log rejection in memory
                if self.memory is not None:
                    if not hasattr(self.memory, 'rejection_log'):
                        self.memory.rejection_log = []
                    self.memory.rejection_log.append({
                        "hypothesis": h,
                        "violations": h.violation_log
                    })
        
        return self.verified_hypotheses

    def learn(self):
        """
        Learning: Update regime beliefs and agent memory using EM.
        Maps to: EM + belief update (agent-controlled learning)
        """
        # Initialize belief state if needed
        if self.belief is None:
            from .belief import BeliefState
            num_regimes = self.config.get("agent", {}).get("num_regimes", 3)
            self.belief = BeliefState(num_regimes=num_regimes)
        
        # Update beliefs based on hypothesis scores (Bayesian)
        self.belief.update(self.verified_hypotheses)
        
        # Store verified hypotheses in memory
        if self.memory is not None:
            for h in self.verified_hypotheses:
                self.memory.store(h)
            
            # Prune invalid hypotheses (ablatable)
            if self.config.get("agent", {}).get("use_memory", True):
                pruned_count = self.memory.prune(current_iteration=self.iteration)
            # else: Ablation - no pruning, memory grows unbounded
        
        return self.belief

    def step(self, data):
        """
        Single GRAIL-V cycle: observe -> retrieve -> reason -> verify -> learn
        
        Config-based ablation flags:
        - use_verification: Toggle physics-based verification
        - use_memory: Toggle memory pruning (in learn())
        - use_belief: Toggle Bayesian belief updates
        - use_reasoning: Toggle symbolic reasoning
        """
        # 1. Perception
        self.observe(data)
        
        # 2. Retrieval
        self.retrieve()
        
        # 3. Reasoning (ablatable)
        if self.config.get("agent", {}).get("use_reasoning", True):
            self.reason()
        else:
            # Ablation: Skip symbolic generation
            self.proposed_hypotheses = []
        
        # 4. Verification (ablatable)
        if self.config.get("agent", {}).get("use_verification", True):
            self.verify()
        else:
            # Ablation: Accept all hypotheses without verification
            self.verified_hypotheses = self.proposed_hypotheses
        
        # 5. Learning (ablatable belief updates, memory handled in learn())
        if self.config.get("agent", {}).get("use_belief", True):
            self.learn()
        else:
            # Ablation: Skip belief updates, keep uniform
            if self.belief is None:
                from .belief import BeliefState
                num_regimes = self.config.get("agent", {}).get("num_regimes", 3)
                self.belief = BeliefState(num_regimes=num_regimes)
            # Store hypotheses but don't update beliefs
            if self.memory is not None:
                for h in self.verified_hypotheses:
                    self.memory.store(h)
        
        self.iteration += 1

    def run_loop(self, data_loader, max_iters=10, tol=1e-3):
        """
        Main agent training loop with autonomous convergence.
        
        Args:
            data_loader: Iterator yielding data batches
            max_iters: Maximum number of iterations
            tol: Convergence tolerance for belief change
        """
        print("Initializing GRAIL-V Agent...")
        import numpy as np
        prev_pi = None
        prev_hypotheses = None
        
        for i in range(max_iters):
            print(f"\n--- Agent Iteration {i+1}/{max_iters} ---")
            
            # Get next data batch
            try:
                data = next(data_loader)
            except StopIteration:
                print("[!] Data exhausted")
                break
            
            # Execute one agent step
            self.step(data)
            
            # Get current state
            pi = self.belief.pi if self.belief else None
            current_eqs = [str(h.equation) for h in self.memory.hypotheses] if self.memory else []
            
            # Report progress
            print(f"  Proposed: {len(self.proposed_hypotheses)}, "
                  f"Verified: {len(self.verified_hypotheses)}, "
                  f"Rejected: {len(self.proposed_hypotheses) - len(self.verified_hypotheses)}")
            
            if pi is not None:
                print(f"  Belief state: {pi}")
            
            # Check belief convergence
            if prev_pi is not None and pi is not None:
                delta = np.linalg.norm(pi - prev_pi)
                print(f"  Belief change: {delta:.6f}")
                
                if delta < tol:
                    print(f"  [✓] Beliefs converged")
                    
                    # Check hypothesis stability
                    if prev_hypotheses is not None:
                        unchanged = set(current_eqs) == set(prev_hypotheses)
                        if unchanged:
                            print(f"\n[CONVERGED] Belief + hypothesis set stabilized at iteration {i+1}")
                            break
            
            prev_pi = pi.copy() if pi is not None else None
            prev_hypotheses = current_eqs
        
        print("\n[+] Agent loop complete")
    
    def introspect(self):
        """
        Agent self-reporting for transparency and debugging.
        
        Returns:
            dict: Current agent state
        """
        top_eqs = []
        if self.memory and self.memory.hypotheses:
            top_eqs = [str(h.equation) for h in self.memory.hypotheses[:5]]
        
        return {
            "iteration": self.iteration,
            "belief": self.belief.pi.tolist() if self.belief else None,
            "num_hypotheses": len(self.memory.hypotheses) if self.memory else 0,
            "num_rejections": len(self.memory.rejection_log) if self.memory and hasattr(self.memory, 'rejection_log') else 0,
            "num_pruned": len(self.memory.pruned_log) if self.memory and hasattr(self.memory, 'pruned_log') else 0,
            "top_equations": top_eqs
        }
