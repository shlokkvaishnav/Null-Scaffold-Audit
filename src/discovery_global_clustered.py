import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from pysr import PySRRegressor

# --- CONFIGURATION ---
DATA_PATH = Path("data/03_processed/training_set.parquet")
N_CLUSTERS = 6  # INCREASED from 3 to 6 (More distinct physics zones)
SUB_SAMPLE = 3000 # Slightly more data per cluster

def run_global_regime_discovery():
    print("🌍 LOADING GLOBAL DATA (ITERATION 2: HIGH COMPLEXITY)...")
    df = pd.read_parquet(DATA_PATH)
    
    # 1. Feature Engineering
    df['Season_Sin'] = np.sin(2 * np.pi * (df['Year'] % 1))
    df['Season_Cos'] = np.cos(2 * np.pi * (df['Year'] % 1))
    
    # 2. Clustering (Now including LONGITUDE)
    # This allows the AI to distinguish the Atlantic from the Pacific
    print(f"\n🧠 CLUSTERING INTO {N_CLUSTERS} REGIMES (SST + Lat + Lon)...")
    clustering_data = df[['SST', 'lat', 'lon']].copy()
    clustering_data['abs_lat'] = clustering_data['lat'].abs()
    
    # Normalize helps K-Means treat Lat/Lon equally
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df['Regime'] = kmeans.fit_predict(clustering_data[['SST', 'abs_lat', 'lon']])
    
    print(f"   Regime Counts: {df['Regime'].value_counts().to_dict()}")

    # 3. Discovery Loop
    regime_models = {}
    
    for i in range(N_CLUSTERS):
        print(f"\n🔬 REGIME {i} (Discovery)")
        regime_df = df[df['Regime'] == i]
        
        # Subsample
        if len(regime_df) > SUB_SAMPLE:
            train_df = regime_df.sample(n=SUB_SAMPLE, random_state=42)
        else:
            train_df = regime_df
            
        X = train_df[["SST", "Salinity", "Year", "Season_Sin", "Season_Cos"]]
        y = train_df["fCO2"]
        
        # PySR (Aggressive search)
        model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*"],
            unary_operators=["sin", "cos"],
            maxsize=20,
            verbosity=0,
            random_state=42
        )
        model.fit(X, y)
        print(f"   Eq: {model.sympy()}")
        regime_models[i] = model

    # 4. Evaluation
    print("\n🌎 EVALUATING HYBRID AGENT...")
    df['Global_Pred'] = np.nan
    for i in range(N_CLUSTERS):
        mask = df['Regime'] == i
        if mask.sum() > 0:
            df.loc[mask, 'Global_Pred'] = regime_models[i].predict(
                df.loc[mask, ["SST", "Salinity", "Year", "Season_Sin", "Season_Cos"]]
            )
            
    final_r2 = r2_score(df['fCO2'], df['Global_Pred'])
    print(f"✅ FINAL GLOBAL R² (K=6): {final_r2:.4f}")

    # 5. Map
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df.sample(5000), x='lon', y='lat', hue='Regime', palette='tab10', s=15)
    plt.title(f"AI Physics Regimes (K=6, R²={final_r2:.2f})")
    plt.savefig("global_regimes_map_v2.png")
    print("🗺️  Map saved to global_regimes_map_v2.png")

if __name__ == "__main__":
    run_global_regime_discovery()