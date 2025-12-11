import yaml
import pandas as pd
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- Configuration ---
DATA_PATH = Path("data/03_processed/training_set.parquet")
CONFIG_PATH = Path("configs/pysr_config.yaml")
RESULTS_DIR = Path("experiments/north_atlantic")

# Regional Box: North Atlantic (approx bounds)
LAT_MIN, LAT_MAX = 20, 60
LON_MIN, LON_MAX = -80, -10

def run_regional_discovery():
    print("🔬 Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    
    # 1. Filter for North Atlantic Region
    #    (Latitude 20N to 60N, Longitude 80W to 10W)
    print(f"   Filtering Region: Lat [{LAT_MIN}, {LAT_MAX}], Lon [{LON_MIN}, {LON_MAX}]")
    regional_df = df[
        (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX) &
        (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
    ].copy()
    
    print(f"   Regional Data Points: {len(regional_df):,}")

    # 2. Subsample if necessary (keep it under 15k for speed)
    if len(regional_df) > 15000:
        regional_df = regional_df.sample(n=15000, random_state=42)

    # 3. Inputs (Physics only)
    #    In a local region, we don't need Lat/Lon as much.
    #    We focus on Thermodynamics (SST) and Chemistry (Salinity) and Trend (Year)
    X = regional_df[["SST", "Salinity", "Year"]]
    y = regional_df["fCO2"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Load Config & Update for Precision
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    # Increase populations slightly for better local search
    config["populations"] = 30 
    
    print("\n🧠 Initializing PySR (Regional Specialist)...")
    model = PySRRegressor(**config)

    print("🚀 Starting North Atlantic Discovery...")
    model.fit(X_train, y_train)

    print("\n" + "="*40)
    print("🏆 NORTH ATLANTIC EQUATION")
    print("="*40)
    print(model.sympy())
    
    r2 = model.score(X_test, y_test)
    print(f"\n✅ Validation R^2 Score: {r2:.4f}")
    
    if r2 > 0.5:
        print("   (BOOM! Now that is a publishable result.)")

if __name__ == "__main__":
    run_regional_discovery()