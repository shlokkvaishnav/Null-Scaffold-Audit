# What is genuinely new vs composition

Use this subsection in method positioning and release-validation notes.

## Why this is not "PySR + MoE glue"
The core novelty is the coupling between:
1. **VSB posterior over symbolic hypotheses** conditioned on regime,
2. **IGBU geodesic regime update** with bounded simplex dynamics,
3. **closed-loop constraint/stability screening** that alters posterior mass and belief flow.

This yields a single inference-and-selection system rather than a stacked pipeline.

## What uncertainty is quantified
- Regime uncertainty: \(\pi_t\) over latent regimes.
- Equation uncertainty: \(q_k(h)\) over symbolic candidates per regime.
- Decision uncertainty: variance across seeds and confidence intervals over metrics.

## Sensitivity to K
Report controlled sweeps over \(K\in\{2,3,4,5\}\) with:
- predictive error,
- calibration error,
- symbolic complexity,
- regime entropy and collapse rate.

## When symbolic models win/lose
- **Win**: low-to-moderate noise, regime-structured dynamics, requirement for interpretable invariants.
- **Lose**: very high-dimensional feature interactions with weak symbolic structure.
- Compare explicitly against neural MoE and boosted trees.
