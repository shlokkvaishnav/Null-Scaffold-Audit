# Training Algorithm: VSB + IGBU (Method Specification)

## Inputs
- Dataset \(\mathcal{D}=\{(x_t,y_t)\}_{t=1}^T\)
- Regime count \(K\)
- Candidate-bank capacity per regime \(N\)
- Gate parameters \(\theta_g\), symbolic search budget \(B\)
- Weights \(\lambda_{gate},\lambda_{KL},\lambda_{stab},\lambda_{phys},\lambda_{complex}\)
- IGBU step size \(\eta\)

## Outputs
- Regime beliefs \(\pi_{1:T}\)
- Regime-conditioned symbolic banks \(\{\mathcal{H}_k\}_{k=1}^K\)
- Final selected equations and uncertainty metrics

## Objective decomposition

\[
\mathcal{L}_{total} = \mathcal{L}_{recon}
+ \lambda_{gate}\mathcal{L}_{gate}
+ \lambda_{KL}\mathcal{L}_{KL}
+ \lambda_{stab}\mathcal{L}_{stab}
+ \lambda_{phys}\mathcal{L}_{phys}
+ \lambda_{complex}\mathcal{L}_{complex}
\]

where:
- \(\mathcal{L}_{recon}\): fit error,
- \(\mathcal{L}_{gate}\): sparse symbolic gate penalty,
- \(\mathcal{L}_{KL}\): temporal / variational regularizer,
- \(\mathcal{L}_{stab}\): Lyapunov/stability penalty,
- \(\mathcal{L}_{phys}\): constraints penalty,
- \(\mathcal{L}_{complex}\): symbolic complexity penalty.

## Algorithm (E/M style)

1. **E-step (routing + latent responsibilities)**
   - compute soft gate assignments \(q_t(z=k)\)
   - derive regime evidence from current symbolic banks

2. **M-step (symbolic hypothesis refinement)**
   - run regime-conditioned symbolic search
   - verify constraints and Lyapunov screening
   - update top-\(N\) bank per regime via VSB scores

3. **Belief dynamics update (IGBU)**
   - compute target \(\tilde\pi_t\) from evidence
   - apply geodesic update:
     \[
     \pi_{t+1}(k) \propto \pi_t(k)^{1-\eta} \tilde\pi_t(k)^\eta
     \]

4. **Parameter update**
   - optimize differentiable parameters with Adam/SGD using \(\mathcal{L}_{total}\)

## Surrogate approximations
- Finite candidate bank \(N\) approximates full symbolic posterior support.
- Relaxed soft responsibilities approximate hard regime assignments.
- Stabilized log-likelihood and clipped probabilities for numerical robustness.

## Complexity
- Symbolic search/evaluation: \(\mathcal{O}(T \cdot K \cdot N \cdot B)\)
- Gate + variational updates: \(\mathcal{O}(T \cdot K)\)
- Belief geodesic updates (IGBU): \(\mathcal{O}(T \cdot K)\)
- Total dominant term: \(\mathcal{O}(TKNB)\)
