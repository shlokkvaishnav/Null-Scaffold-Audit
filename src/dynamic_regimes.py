import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from pathlib import Path

# --- CONFIGURATION ---
DATA_PATH = Path("data/03_processed/training_set.parquet")
LOCATION_LAT = 30.0  # Approx Bermuda (North Atlantic)
LOCATION_LON = -60.0
N_STATES = 2         # e.g., "Winter Mode" vs "Summer Mode"

def run_dynamic_regime_detection():
    print("⏳ LOADING DATA FOR DYNAMIC HMM ANALYSIS...")
    df = pd.read_parquet(DATA_PATH)
    
    # 1. Filter for a specific time-series (One location)
    # We grab points close to our target lat/lon to simulate a single sensor station
    local_df = df[
        (df['lat'].between(LOCATION_LAT - 2, LOCATION_LAT + 2)) & 
        (df['lon'].between(LOCATION_LON - 2, LOCATION_LON + 2))
    ].sort_values('Year').copy()
    
    if len(local_df) < 50:
        print("❌ Not enough data for this location. Try different coordinates.")
        return

    print(f"   Analyzing {len(local_df)} months of data at Lat {LOCATION_LAT}, Lon {LOCATION_LON}...")

    # 2. Prepare Data for HMM
    # The HMM looks at [SST, fCO2] to decide what 'state' the ocean is in
    X = local_df[['SST', 'fCO2']].values

    # 3. Train Gaussian HMM
    # This finds 'hidden states' (e.g., Upwelling vs. Stratified) that generate the data
    print(f"🧠 Training HMM with {N_STATES} hidden states...")
    model = hmm.GaussianHMM(n_components=N_STATES, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X)
    
    # Predict the hidden state for each month
    hidden_states = model.predict(X)
    local_df['State'] = hidden_states

    # 4. Visualize the Dynamic Transitions
    print("📈 Plotting Regime Transitions...")
    plt.figure(figsize=(12, 6))
    
    # Plot Temperature, colored by the Hidden State
    # State 0 = Blue, State 1 = Red
    colors = ['blue' if s == 0 else 'red' for s in hidden_states]
    
    plt.scatter(local_df['Year'], local_df['fCO2'], c=colors, s=15, alpha=0.7)
    plt.plot(local_df['Year'], local_df['fCO2'], c='gray', alpha=0.3, linewidth=1)
    
    plt.title(f"Dynamic Regime Switching Detected by HMM (Lat {LOCATION_LAT})")
    plt.xlabel("Year")
    plt.ylabel("Ocean fCO2")
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Regime A (e.g., Winter/Sink)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Regime B (e.g., Summer/Source)')
    ]
    plt.legend(handles=legend_elements)
    
    plt.savefig("dynamic_regime_hmm.png")
    print("✅ Analysis Complete. Saved plot to 'dynamic_regime_hmm.png'")
    
    # 5. Print Transition Matrix (The "Physics" part)
    print("\n🔄 Regime Transition Matrix (Probability of switching states):")
    print(model.transmat_.round(3))
    print("\n(Example: Row 0, Col 1 is probability of switching from Regime A -> Regime B)")

if __name__ == "__main__":
    run_dynamic_regime_detection()