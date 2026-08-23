# NAS RandomSearch self-audit: margin-sensitivity sweep

Source: `results\nas_search_self_audit\audit.json` (issue #11 / PR #12), n=30 paired `valid_accuracy` observations, held fixed.

90% BCa bootstrap CI on the paired difference (higher_is_better=True, resamples=10000, seed=0): [-0.1970, -0.0081] percentage points.

Pre-registered margin (issue #11): +/-0.3pp -> verdict **NULL** (matches PR #12's reported verdict).

## Crossovers

- HARMFUL -> INCONCLUSIVE at margin = 0.0081pp
- INCONCLUSIVE -> NULL at margin = 0.1970pp

Nearest crossover to the pre-registered margin: 0.1970pp (INCONCLUSIVE <-> NULL), a distance of 0.1030pp (1.52x).
