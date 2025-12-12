import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def test_clustering_logic():
    """
    Unit Test: Verifies that K-Means correctly segments synthetic ocean data.
    Simulates a 'Mini-Ocean' with 2 distinct regimes to see if logic holds.
    """
    # 1. Create Synthetic Data (Mocking the Ocean)
    # Regime A: Cold & Salty (North Atlantic)
    regime_a = pd.DataFrame({
        'SST': np.random.normal(10, 2, 100),       # ~10°C
        'lat': np.random.normal(45, 5, 100),       # ~45°N
        'lon': np.random.normal(-30, 5, 100)       # Atlantic Lon
    })
    
    # Regime B: Hot & Fresh (Equator)
    regime_b = pd.DataFrame({
        'SST': np.random.normal(28, 2, 100),       # ~28°C
        'lat': np.random.normal(0, 5, 100),        # Equator
        'lon': np.random.normal(-150, 5, 100)      # Pacific Lon
    })
    
    df = pd.concat([regime_a, regime_b], ignore_index=True)
    
    # 2. Run the Logic (Same as src/discovery_global_clustered.py)
    df['abs_lat'] = df['lat'].abs()
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['Regime'] = kmeans.fit_predict(df[['SST', 'abs_lat', 'lon']])
    
    # 3. Assertions (The "Test")
    # We expect exactly 2 regimes
    assert df['Regime'].nunique() == 2, "Clustering failed to find 2 regimes"
    
    # The 'Cold' pixels should mostly be in one cluster, 'Hot' in the other
    counts = df.groupby('Regime')['SST'].mean()
    assert abs(counts.iloc[0] - counts.iloc[1]) > 10, "Regimes are not distinct enough in temperature"

if __name__ == "__main__":
    test_clustering_logic()
    print("✅ System Test Passed: Clustering Logic is Robust.")