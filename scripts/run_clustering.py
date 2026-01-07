import xarray as xr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIG ---
DATA_PATH = Path("data/03_processed/climate_fused_dataset.nc")
OUTPUT_IMG = Path("figures/kmeans_regimes.png")
FEATURES = ['sst', 'sss', 'sin_month', 'cos_month', 'log_chl'] # Clustering on physical features
N_CLUSTERS = 6  # Standard number of ocean biomes

def main():
    # 1. Load Data
    if not DATA_PATH.exists():
        print("❌ Data not found.")
        return
        
    ds = xr.open_dataset(DATA_PATH)
    df = ds.to_dataframe().reset_index().dropna(subset=FEATURES)
    
    # 2. Normalize Data (Crucial for K-Means)
    print("🔄 Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    
    # 3. Fit K-Means
    print(f"🧩 Clustering into {N_CLUSTERS} regimes...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df['regime'] = kmeans.fit_predict(X_scaled)
    
    # 4. Visualize the Regimes (The "Map")
    print("🗺️  Generating regime map...")
    plt.figure(figsize=(12, 6))
    
    # Plot only a subset of points for speed/clarity
    subset = df.sample(n=min(50000, len(df)), random_state=42)
    
    sns.scatterplot(
        data=subset, 
        x='lon', 
        y='lat', 
        hue='regime', 
        palette='viridis', 
        s=5, 
        marker='s',
        edgecolor='none'
    )
    
    plt.title(f"Hard K-Means Ocean Regimes (K={N_CLUSTERS})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    
    # Save
    OUTPUT_IMG.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"✅ Map saved to {OUTPUT_IMG}")
    
    # 5. Interpretability Check: What defines each regime?
    print("\n📊 Regime Statistics (Mean Values):")
    stats = df.groupby('regime')[FEATURES].mean()
    print(stats)

if __name__ == "__main__":
    main()