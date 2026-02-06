"""
Agent Orchestrator.
Manages the Perception -> Retrieval -> Reasoning -> Verification -> Learning loop.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Iterator


class SDMoSEAgent:
    """
    Agent Orchestrator.
    Manages the Perception -> Retrieval -> Reasoning -> Verification -> Learning loop.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SD-MoSE agent.
        
        Args:
            config: Configuration dictionary with agent settings
        """
        self.config = config or {}
        
        # Core components (lazy initialized)
        self.perception = None
        self.retrieval_module = None
        self.reasoning_module = None
        self.verification_module = None
        self.memory = None
        self.belief = None
        
        # State
        self.observation: Optional[Dict] = None
        self.priors: Optional[Dict] = None
        self.candidate_hypotheses: List = []
        self.proposed_hypotheses: List = []
        self.verified_hypotheses: List = []
        self.iteration: int = 0
        
        # Cache config values
        self._num_regimes = self.config.get("agent", {}).get("num_regimes", 3)

    def observe(self, data: Any) -> Dict:
        """
        Perception: Encode raw data into agent observations.
        
        Args:
            data: Raw data (dict with features/targets or numpy array)
            
        Returns:
            dict: Structured observation
        """
        if self.perception is None:
            from .perception import PerceptionModule
            self.perception = PerceptionModule(self.config)
        
        self.observation = self.perception.encode(data)
        return self.observation

    def retrieve(self) -> tuple:
        """
        Retrieval: Fetch relevant priors and historical regime patterns.
        
        Returns:
            tuple: (priors dict, candidate hypotheses list)
        """
        if self.retrieval_module is None:
            from .retrieval import RetrievalModule
            self.retrieval_module = RetrievalModule(self.config)
        
        self.priors = self.retrieval_module.retrieve_priors()
        self.candidate_hypotheses = self.retrieval_module.retrieve_hypotheses(self.memory)
        
        return self.priors, self.candidate_hypotheses

    def reason(self) -> List:
        """
        Reasoning: Propose symbolic hypotheses for each regime.
        
        Returns:
            list: Proposed Hypothesis objects
        """
        if self.reasoning_module is None:
            from .reasoning import ReasoningModule
            self.reasoning_module = ReasoningModule(self.config)
        
        self.proposed_hypotheses = []
        
        for k in range(self._num_regimes):
            h = self.reasoning_module.propose_hypothesis(
                observation=self.observation,
                regime_id=k,
                priors=self.priors
            )
            if h is not None:
                self.proposed_hypotheses.append(h)
        
        return self.proposed_hypotheses

    def verify(self) -> List:
        """
        Verification: Validate hypotheses against physics constraints.
        
        Returns:
            list: Verified Hypothesis objects
        """
        if self.verification_module is None:
            from .verification import VerificationModule
            self.verification_module = VerificationModule(self.config)
        
        # Initialize memory if needed (for rejection logging)
        if self.memory is None:
            from .memory import AgentMemory
            self.memory = AgentMemory()
        
        self.verified_hypotheses = []
        
        for h in self.proposed_hypotheses:
            # Verify against physics constraints
            h.verify(self.verification_module)
            
            # Score hypothesis with data fit
            self.verification_module.score_hypothesis(h, self.observation)
            
            if h.valid:
                self.verified_hypotheses.append(h)
            else:
                # Log rejection
                self.memory.log_rejection(h)
        
        return self.verified_hypotheses

    def learn(self) -> Any:
        """
        Learning: Update regime beliefs and agent memory.
        
        Returns:
            BeliefState: Updated belief state
        """
        # Initialize memory if needed
        if self.memory is None:
            from .memory import AgentMemory
            self.memory = AgentMemory()
        
        # Initialize belief state if needed
        if self.belief is None:
            from .belief import BeliefState
            self.belief = BeliefState(num_regimes=self._num_regimes)
        
        # Update beliefs based on hypothesis scores
        self.belief.update(self.verified_hypotheses)
        
        # Store verified hypotheses in memory
        for h in self.verified_hypotheses:
            self.memory.store(h)
        
        # Prune memory (ablatable)
        if self.config.get("agent", {}).get("use_memory", True):
            self.memory.prune(current_iteration=self.iteration)
        
        return self.belief

    def step(self, data: Any) -> None:
        """
        Single agent cycle: observe -> retrieve -> reason -> verify -> learn
        
        Config-based ablation flags:
        - use_verification: Toggle physics-based verification
        - use_memory: Toggle memory pruning (in learn())
        - use_belief: Toggle Bayesian belief updates
        - use_reasoning: Toggle symbolic reasoning
        
        Args:
            data: Input data for this iteration
        """
        agent_config = self.config.get("agent", {})
        
        # 1. Perception
        self.observe(data)
        
        # 2. Retrieval
        self.retrieve()
        
        # 3. Reasoning (ablatable)
        if agent_config.get("use_reasoning", True):
            self.reason()
        else:
            self.proposed_hypotheses = []
        
        # 4. Verification (ablatable)
        if agent_config.get("use_verification", True):
            self.verify()
        else:
            # Accept all hypotheses without verification
            self.verified_hypotheses = self.proposed_hypotheses.copy()
            # Still score them for belief updates
            if self.verification_module is None:
                from .verification import VerificationModule
                self.verification_module = VerificationModule(self.config)
            for h in self.verified_hypotheses:
                self.verification_module.score_hypothesis(h, self.observation)
        
        # 5. Learning (ablatable)
        if agent_config.get("use_belief", True):
            self.learn()
        else:
            # Initialize belief without updating
            if self.belief is None:
                from .belief import BeliefState
                self.belief = BeliefState(num_regimes=self._num_regimes)
            # Store hypotheses without belief updates
            if self.memory is None:
                from .memory import AgentMemory
                self.memory = AgentMemory()
            for h in self.verified_hypotheses:
                self.memory.store(h)
        
        self.iteration += 1

    def run_loop(self, data_loader: Iterator, max_iters: int = 10, tol: float = 1e-3) -> None:
        """
        Main agent training loop with autonomous convergence.
        
        Args:
            data_loader: Iterator yielding data batches
            max_iters: Maximum number of iterations
            tol: Convergence tolerance for belief change
        """
        print("Initializing SD-MoSE Agent...")
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
            current_eqs = self._get_current_equations()
            
            # Report progress
            self._report_progress(i, pi, prev_pi, tol)
            
            # Check convergence
            if self._check_convergence(pi, prev_pi, current_eqs, prev_hypotheses, tol, i):
                break
            
            prev_pi = pi.copy() if pi is not None else None
            prev_hypotheses = current_eqs
        
        print("\n[+] Agent loop complete")
    
    def _get_current_equations(self) -> List[str]:
        """Get list of equations currently in memory."""
        if self.memory and self.memory.hypotheses:
            return [str(h.equation) for h in self.memory.hypotheses]
        return []
    
    def _report_progress(self, iteration: int, pi: Optional[np.ndarray], 
                         prev_pi: Optional[np.ndarray], tol: float) -> None:
        """Report iteration progress."""
        n_proposed = len(self.proposed_hypotheses)
        n_verified = len(self.verified_hypotheses)
        n_rejected = n_proposed - n_verified
        
        print(f"  Proposed: {n_proposed}, Verified: {n_verified}, Rejected: {n_rejected}")
        
        if pi is not None:
            print(f"  Belief state: {pi}")
            
            if prev_pi is not None:
                delta = np.linalg.norm(pi - prev_pi)
                print(f"  Belief change: {delta:.6f}")
    
    def _check_convergence(self, pi: Optional[np.ndarray], prev_pi: Optional[np.ndarray],
                           current_eqs: List[str], prev_hypotheses: Optional[List[str]],
                           tol: float, iteration: int) -> bool:
        """Check if agent has converged."""
        if prev_pi is None or pi is None:
            return False
        
        delta = np.linalg.norm(pi - prev_pi)
        
        if delta < tol:
            print("  [+] Beliefs converged")
            
            if prev_hypotheses is not None and set(current_eqs) == set(prev_hypotheses):
                print(f"\n[CONVERGED] Belief + hypothesis set stabilized at iteration {iteration+1}")
                return True
        
        return False
    
    def introspect(self) -> Dict[str, Any]:
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
            "num_rejections": self.memory.get_rejection_count() if self.memory else 0,
            "num_pruned": self.memory.get_pruned_count() if self.memory else 0,
            "top_equations": top_eqs
        }
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        self.memory = None
        self.belief = None
        self.observation = None
        self.priors = None
        self.candidate_hypotheses = []
        self.proposed_hypotheses = []
        self.verified_hypotheses = []
        self.iteration = 0
