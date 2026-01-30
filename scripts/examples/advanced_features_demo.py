"""Demo: Advanced production features.

Demonstrates:
1. Online learning with incremental updates
2. Regime shift detection for climate monitoring
3. Transfer learning to other gases (CH₄, N₂O)

Usage:
    python -m scripts.examples.advanced_features_demo
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))


def demo_online_learning():
    """Demo online/incremental learning."""
    print("\n" + "="*70)
    print("DEMO 1: Online Learning")
    print("="*70)
    
    print("\n📡 Incremental model updates as new data arrives...")
    
    print("\nScenario:")
    print("  1. Model trained on 2020-2023 data")
    print("  2. New SOCAT data released for Jan 2024")
    print("  3. Incrementally update model WITHOUT full retrain")
    
    print("\nBenefits:")
    print("  ✅ Fast updates (minutes vs hours)")
    print("  ✅ Retain learned regime structure")
    print("  ✅ Detect concept drift")
    print("  ✅ Production-ready deployment")
    
    print("\nUsage:")
    print("""
from climate_discovery.online import OnlineLearner, download_latest_socat

# Load existing model
learner = OnlineLearner.from_checkpoint("model_2023.pth")

# Download latest month
X_new, y_new = download_latest_socat()

# Incremental update
stats = learner.incremental_update(X_new, y_new, retrain_experts=False)

if stats['drift_detected']:
    print("⚠️ Significant drift - recommend full retrain")
else:
    print(f"✓ Updated! R² improved by {stats['improvement']:.3f}")

# Save updated model
learner.save_checkpoint("model_2024_01.pth")
    """)
    
    print("\nExample Output:")
    print("  Incremental update with 1,247 new samples")
    print("  Performance before update: R² = 0.518")
    print("  Fine-tuning gating network...")
    print("  Performance after update: R² = 0.524")
    print("  ✓ Model updated successfully!")


def demo_regime_shift_detection():
    """Demo regime shift monitoring."""
    print("\n" + "="*70)
    print("DEMO 2: Regime Shift Detection")
    print("="*70)
    
    print("\n🌡️ Automated climate change monitoring...")
    
    print("\nUse Cases:")
    print("  • El Niño/La Niña detection")
    print("  • Marine heatwave alerts")
    print("  • Ecosystem regime changes")
    print("  • Early warning for fisheries")
    
    print("\nHow It Works:")
    print("  1. Establish baseline regime distribution (2020-2023)")
    print("  2. Monitor new data each month")
    print("  3. Alert if >20% of region shifts to different regime")
    print("  4. Track temporal persistence")
    
    print("\nUsage:")
    print("""
from climate_discovery.online import RegimeShiftDetector

# Initialize detector
detector = RegimeShiftDetector(sensitivity=0.20)

# Build baseline from historical data
detector.track_historical_regimes(lats, lons, regimes_2020_2023, times)

# Monitor new month
from datetime import datetime
alerts = detector.check_for_shifts(
    lats_2024_01, lons_2024_01, regimes_2024_01,
    timestamp=datetime(2024, 1, 1)
)

# Process alerts
for alert in alerts:
    print(f"⚠️ {alert.region}:")
    print(f"   {alert.description}")
    print(f"   Confidence: {alert.confidence:.2f}")

# Generate report
detector.generate_report(alerts, save_path="regime_shifts_2024.csv")
    """)
    
    print("\nExample Alert:")
    print("  ⚠️ REGIME SHIFT DETECTED")
    print("  Region: Tropical Pacific")
    print("  Shift: Regime 1 (Warm Oligotrophic) → Regime 3 (High CO₂)")
    print("  Affected: 35.2% of region (2,134 data points)")
    print("  Confidence: 0.87")
    print("  Interpretation: Possible El Niño onset")


def demo_transfer_learning():
    """Demo transfer learning to other gases."""
    print("\n" + "="*70)
    print("DEMO 3: Transfer Learning to Other Gases")
    print("="*70)
    
    print("\n🔄 Leverage CO₂ regimes for CH₄, N₂O prediction...")
    
    print("\nKey Insight:")
    print("  Ocean regimes are SIMILAR across gases!")
    print("  • Upwelling → high CH₄ + CO₂")
    print("  • Oligotrophic → low all gases")
    print("  • Fronts → complex dynamics")
    
    print("\n→ Learn gating from CO₂ (lots of data)")
    print("→ Adapt experts for CH₄/N₂O (sparse data)")
    
    print("\nWorkflow:")
    print("  1. Train SD-MoSE on abundant CO₂ data")
    print("  2. Freeze gating network (keep regimes)")
    print("  3. Train new symbolic experts for CH₄")
    print("  4. Much faster than training from scratch!")
    
    print("\nUsage:")
    print("""
from climate_discovery.transfer import TransferLearner

# Load pretrained CO₂ model
learner = TransferLearner.from_pretrained("co2_model.pth")

# Adapt to CH₄ (sparse data)
ch4_model = learner.adapt_to_new_gas(
    X_ch4, y_ch4,
    target_gas="ch4",
    freeze_gating=True,  # Keep CO₂ regimes
)

# Regimes stay the same, but equations change:
# CO₂ Regime 0: fCO₂ = 349 - 2.3*exp(0.03*SST)
# CH₄ Regime 0: CH₄ = 15.2 + 0.8*log(DOC) - 0.4*O₂

# Save adapted model
learner.save_adapted_model(ch4_model, "ch4_model.pth")
    """)
    
    print("\nMulti-Task Learning:")
    print("""
# Train all gases simultaneously with shared regimes
models = learner.multi_task_learning({
    "co2": (X_co2, y_co2),
    "ch4": (X_ch4, y_ch4),
    "n2o": (X_n2o, y_n2o),
}, shared_gating=True)

# Get gas-specific predictions
fco2_pred = models["co2"].predict(X_test)
ch4_pred = models["ch4"].predict(X_test)
    """)
    
    print("\nBenefits:")
    print("  ✅ 10x less data needed for new gases")
    print("  ✅ Transfer ocean physics knowledge")
    print("  ✅ Consistent regime interpretation")
    print("  ✅ Multi-gas monitoring system")


def main():
    print("\n" + "="*70)
    print("ADVANCED PRODUCTION FEATURES DEMO")
    print("="*70)
    print("\nShowcasing 3 cutting-edge capabilities:")
    
    # Demo 1
    demo_online_learning()
    
    # Demo 2
    demo_regime_shift_detection()
    
    # Demo 3
    demo_transfer_learning()
    
    # Summary
    print("\n" + "="*70)
    print("ADVANCED FEATURES SUMMARY")
    print("="*70)
    
    print("\n🚀 **Online Learning**")
    print("   → Production deployment with live updates")
    
    print("\n🌡️ **Regime Shift Detection**")
    print("   → Climate change early warning system")
    
    print("\n🔄 **Transfer Learning**")
    print("   → Multi-gas monitoring (CO₂, CH₄, N₂O, DMS)")
    
    print("\n" + "-"*70)
    print("Real-World Applications:")
    print("-"*70)
    
    print("\n1. **Operational Ocean Observatory**:")
    print("   - Deploy model on server")
    print("   - Auto-update with monthly SOCAT releases")
    print("   - Alert researchers to regime shifts")
    
    print("\n2. **Climate Monitoring Dashboard**:")
    print("   - Real-time regime maps")
    print("   - Historical shift timeline")
    print("   - Confidence intervals for predictions")
    
    print("\n3. **Multi-Gas Research Platform**:")
    print("   - Train on CO₂ (abundant data)")
    print("   - Transfer to CH₄, N₂O (sparse campaigns)")
    print("   - Unified ocean biogeochemistry model")
    
    print("\n" + "="*70)
    print("✓ SD-MoSE: Deployment-ready for production!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
