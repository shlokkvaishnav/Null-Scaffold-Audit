import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class JointELBO(nn.Module):
    """
    Computes the joint Evidence Lower Bound (ELBO) over both regime
    routing probabilities and optimal symbolic hypothesis scores via
    a temporal Variational sequence tracker.
    
    L = E_q[log p(y|x,z,h)] - KL[q(z) || p(z_t|z_{t-1})] - sum_k KL[q_k(h) || p_k(h)]
    """
    def __init__(self, num_regimes: int, transition_prior: float = 0.9):
        """
        Args:
            num_regimes: Unique discrete active regions
            transition_prior: Baseline persistence probability across temporal sequence (Markov diagonal)
        """
        super().__init__()
        self.num_regimes = num_regimes
        
        # Markov HMM Transition Matrix A_{j,k} representing p(z_t=k | z_{t-1}=j)
        # Initialized heavily on the diagonal for temporal Earth system persistence
        init_A = torch.ones(num_regimes, num_regimes) * ((1.0 - transition_prior) / max(1, num_regimes - 1))
        init_A.fill_diagonal_(transition_prior)
        
        # Stored dynamically as unconstrained logits for gradient descent
        self.transition_logits = nn.Parameter(torch.log(init_A + 1e-8))
        
    def get_transition_matrix(self) -> torch.Tensor:
        """Returns the normalized transition probability matrix A"""
        return F.softmax(self.transition_logits, dim=1)
        
    def forward(self, log_likelihoods: torch.Tensor, gate_probs: torch.Tensor, 
                regime_priors: Optional[torch.Tensor] = None,
                equation_beliefs: Optional[Dict[int, Dict[str, float]]] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate ELBO over sequenced batches.
        
        Args:
            log_likelihoods: (T, Num_Regimes) E_q_k[log p] extracted from hypothesis strings
            gate_probs: (T, Num_Regimes) Soft categorization q(z) dynamically issued from SGF
            regime_priors: Unused parameter kept for compatibility with trainer
            equation_beliefs: Tracked hypothesis probabilities (optional/bridged inside likelihoods now)
        """
        T = gate_probs.size(0)
        
        # 1. Expected Log-Likelihood E_q(z) [ E_q_k(h) [ log p(y | x, z, h) ] ]
        # Assumes log_likelihoods natively already contain the inner hypothesis evaluation expectation
        expected_log_likelihood = torch.sum(gate_probs * log_likelihoods)
        
        # 2. Variational KL Divergence for Temporal HMM Sequences
        A = self.get_transition_matrix()
        
        # Flat starting condition
        prior_0 = torch.ones(self.num_regimes, device=gate_probs.device) / self.num_regimes
        kl_z = torch.sum(gate_probs[0] * torch.log(gate_probs[0] / prior_0 + 1e-12))
        
        if T > 1:
            # Expected prior from immediate previous sequence states: p(z_t) = q(z_{t-1}) @ A
            q_prev = gate_probs[:-1]
            predicted_prior = torch.matmul(q_prev, A)
            
            q_curr = gate_probs[1:]
            kl_z_seq = torch.sum(q_curr * torch.log(q_curr / predicted_prior + 1e-12))
            
            # Combine temporal bounds
            kl_z = kl_z + kl_z_seq
            
        # 3. Hypothesis KL Divergences (optional, if structure maintained as detached distributions)
        kl_h_total = torch.tensor(0.0, device=gate_probs.device)
        
        # We minimize NEGATIVE ELBO
        elbo = expected_log_likelihood - kl_z - kl_h_total
        loss = -elbo
        
        return {"loss": loss, "elbo": elbo, "kl_z": kl_z, "ell": expected_log_likelihood}
