# SD-MoSE Decision Freeze (Project-Specific)

This document locks the paper narrative and implementation targets for the current codebase.

## 1) Contribution Scope (Frozen)

### Primary contribution
**Variational Symbolic Bayes (VSB)** over regime-conditioned equation hypotheses.

### Secondary contribution
**Information-Geometric Belief Update (IGBU)** for regime belief dynamics on the simplex.

### Support components (non-headline)
- Symbolic gate
- Lyapunov screening
- Constraint engine
- Memory loop

## 2) Theorem Target

### Main theorem (required)
**IGBU bounded-divergence / monotone interpolation theorem**.

For source belief \(\pi_t\), target belief \(\tilde\pi\), and \(\eta \in [0,1]\), define:

\[
\pi_{t+1}(k) = \frac{\pi_t(k)^{1-\eta}\,\tilde\pi(k)^\eta}{\sum_j \pi_t(j)^{1-\eta}\,\tilde\pi(j)^\eta}
\]

The update:
1. stays in the simplex interior,
2. interpolates endpoints exactly at \(\eta=0,1\),
3. yields monotone movement toward \(\tilde\pi\) under KL along practical η-schedules.

### Optional proposition
Surrogate ELBO model-selection consistency under finite candidate bank assumptions.

## 3) Camera-Ready Algorithm Requirements

The paper must include one explicit algorithm with:
- inputs / outputs,
- exact objective decomposition,
- E-step / M-step blocks,
- where surrogate approximations are used,
- complexity in \(K\) (regimes), \(T\) (time / iterations), and \(N\) (candidate bank size).

Reference implementation: `docs/training_algorithm_vsb_igbu.md`.

## 4) Baseline Package (Minimum)

- PySR global
- Neural MoE
- LightGBM or XGBoost
- Full SD-MoSE model
- Ablations: `-VSB`, `-IGBU`, `-constraints/stability`

Statistical protocol:
- ≥ 5 seeds (default),
- 10 seeds when compute budget allows,
- confidence intervals + significance tests.

## 5) Reproducibility Bar

Require before submission:
- fixed config files for each figure/table,
- deterministic seed policy,
- one-command benchmark reproduction,
- runtime budget table (small reproducible + full run),
- logged hyperparameter search protocol.

Reference implementation:
- `configs/paper/*.yaml`
- `scripts/reproduce_benchmarks.py`
- `docs/reproducibility_protocol.md`

## 6) Intro / Rebuttal Preemption

Add subsection: **"What is genuinely new vs composition"**, covering:
- why this is not "PySR + MoE glue",
- what uncertainty is quantified and how calibrated,
- sensitivity to \(K\),
- when symbolic models outperform or underperform black-box models.

Reference text template: `docs/novelty_positioning.md`.
