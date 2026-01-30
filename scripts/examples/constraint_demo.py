"""Example demonstrating physics-informed constraints in SD-MoSE.

This script shows how constraints prevent unphysical equations and compares
results with/without constraints.

Usage:
    python -m scripts.examples.constraint_demo
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulated PySR examples (for demonstration)

print("=" * 70)
print("PHYSICS-INFORMED CONSTRAINTS DEMO")
print("=" * 70)
print()

print("WITHOUT Constraints:")
print("-" * 70)
print("Regime 0: fCO2 = 349.2 + exp(247.5 * SST)  ❌ OVERFLOW!")
print("  → exp(247.5 * 20) = exp(4950) → ∞ (numerical instability)")
print()
print("Regime 1: fCO2 = SST^157 * log(-SSS)  ❌ UNPHYSICAL!")
print("  → Insane exponent, log of negative value")
print()
print("Regime 2: fCO2 = sqrt(sqrt(sqrt(log(log(SST)))))  ❌ OVERLY COMPLEX!")
print("  → Deeply nested, no physical meaning")
print()

print("WITH Physics-Informed Constraints:")
print("-" * 70)
print("Regime 0: fCO2 = 349.56 - 2.34 * exp(0.031 * SST)  ✅ STABLE!")
print("  → exp(0.031 * 20) = exp(0.62) = 1.86 (bounded)")
print()
print("Regime 1: fCO2 = 380.2 + 3.14 * SST - 1.57 * SSS  ✅ LINEAR!")
print("  → Simple, interpretable, physically plausible")
print()
print("Regime 2: fCO2 = 412.3 * (1 + 0.045 * log(Chl))  ✅ REALISTIC!")
print("  → Log-linear biology term (standard in lit)")
print()

print("=" * 70)
print("CONSTRAINT SUMMARY")
print("=" * 70)
print()
print("Operator Constraints:")
print("  exp(x):  max_complexity = 5  → prevents exp(complex_expr)")
print("  pow(x,y): y must be constant  → prevents dynamic exponents")
print("  /:  denominator != 0        → prevents division by zero")
print()
print("Nested Constraints:")
print("  exp(exp(x))    ❌  Prevented")
print("  exp(log(x))    ❌  Prevented (simplifies to x)")
print("  log(log(x))    ❌  Prevented")
print("  sqrt(sqrt(x))  ❌  Prevented")
print()
print("Complexity Limits:")
print("  max_tree_depth = 10  → prevents deeply nested expressions")
print("  max_size = 25        → limits total equation nodes")
print("  parsimony = 0.01     → penalizes complexity in Pareto front")
print()

# Generate example plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Exponential with/without constraint
sst_range = np.linspace(-2, 35, 100)

# Unconstrained (would overflow)
try:
    y_unconstrained = 349.2 + np.exp(247.5 * (sst_range - 15) / 10)  # Standardized
    y_unconstrained = np.clip(y_unconstrained, 0, 1e6)  # Clip for visibility
except:
    y_unconstrained = np.full_like(sst_range, np.nan)

# Constrained (stable)
y_constrained = 349.56 - 2.34 * np.exp(0.031 * sst_range)

axes[0].plot(sst_range, y_unconstrained, 'r--', linewidth=2, label='Unconstrained: exp(247.5×SST)', alpha=0.7)
axes[0].plot(sst_range, y_constrained, 'g-', linewidth=2, label='Constrained: exp(0.031×SST)')
axes[0].axhline(y=600, color='k', linestyle=':', alpha=0.5, label='Physical max (~600 μatm)')
axes[0].set_xlabel('SST (°C)', fontsize=12)
axes[0].set_ylabel('fCO₂ (μatm)', fontsize=12)
axes[0].set_title('Exponential Constraints Prevent Overflow', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].set_ylim([250, 700])
axes[0].grid(alpha=0.3)

# Right: Complexity comparison
regimes = ['Regime 0', 'Regime 1', 'Regime 2', 'Regime 3', 'Regime 4', 'Regime 5']
complexity_unconstrained = [47, 35, 28, 52, 41, 39]  # Bloated equations
complexity_constrained = [8, 5, 12, 9, 7, 11]  # Simpler with constraints

x = np.arange(len(regimes))
width = 0.35

axes[1].bar(x - width/2, complexity_unconstrained, width, label='Without Constraints', color='salmon')
axes[1].bar(x + width/2, complexity_constrained, width, label='With Constraints', color='lightgreen')
axes[1].axhline(y=25, color='k', linestyle='--', alpha=0.5, label='Max Size Limit')
axes[1].set_ylabel('Equation Complexity (nodes)', fontsize=12)
axes[1].set_title('Constraints Reduce Overfitting', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(regimes, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/physics_constraints_demo.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved: figures/physics_constraints_demo.png")
print()

# Print scientific rationale
print("=" * 70)
print("SCIENTIFIC RATIONALE")
print("=" * 70)
print()
print("Ocean Carbon Chemistry Bounds:")
print("  • fCO₂ range: ~200-600 μatm (observed in SOCAT)")
print("  • SST range: -2 to 35°C (polar to tropical)")
print("  • SSS range: 0 to 42 PSU (freshwater to hypersaline)")
print("  • Chl range: 0.01 to 100 mg/m³ (oligotrophic to bloom)")
print()
print("Why Constrain exp()?")
print("  • Henry's Law: fCO₂ ∝ exp(a*SST) where a ≈ 0.01-0.05")
print("  • Unconstrained: PySR finds a=247 → numerical explosion")
print("  • Constraint: Limit complexity → forces realistic coefficients")
print()
print("Why Constrain pow()?")
print("  • Physical laws use integer exponents (square, cube)")
print("  • Unconstrained: SST^157 is dimensionally absurd")
print("  • Constraint: Only allow constant exponents (e.g., x^2)")
print()
print("Benefits:")
print("  ✓ Numerical stability (no overflow/NaN)")
print("  ✓ Physically interpretable equations")
print("  ✓ Better generalization (less overfitting)")
print("  ✓ Faster convergence (smaller search space)")
print()
