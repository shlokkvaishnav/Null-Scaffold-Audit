import yaml
import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- Configuration ---
DATA_PATH = Path("data/03_processed/training_set.parquet")
CONFIG_PATH = Path("configs/pysr_config.yaml")

# The "Bermuda" Box (BATS Region)
LAT_MIN, LAT_MAX = 25, 35
LON_MIN, LON_MAX = -70, -60

def run_bats_discovery():
    print("🔬 Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    
    # 1. Filter for Bermuda Region
    print(f"   Targeting Bermuda (BATS): Lat [{LAT_MIN}, {LAT_MAX}], Lon [{LON_MIN}, {LON_MAX}]")
    regional_df = df[
        (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX) &
        (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
    ].copy()
    
    # --- SPEED HACK ---
    if len(regional_df) > 2000:
        print(f"   ⚡ Subsampling {len(regional_df):,} -> 2,000 points for speed...")
        regional_df = regional_df.sample(n=2000, random_state=42)

    # 2. Feature Engineering
    print("   Engineering Seasonal Cycles...")
    regional_df['Season_Sin'] = np.sin(2 * np.pi * (regional_df['Year'] % 1))
    regional_df['Season_Cos'] = np.cos(2 * np.pi * (regional_df['Year'] % 1))

    # 3. Inputs
    X = regional_df[["SST", "Salinity", "Year", "Season_Sin", "Season_Cos"]]
    y = regional_df["fCO2"]

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Load Config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    # Overrides for BATS specific run
    config["populations"] = 40
    config["niterations"] = 300  # <--- FIXED KEYWORD
    
    print("\n🧠 Initializing PySR (Bermuda Specialist)...")
    model = PySRRegressor(**config)

    print("🚀 Starting Discovery Loop...")
    model.fit(X_train, y_train)

    print("\n" + "="*40)
    print("🏆 THE BERMUDA EQUATION")
    print("="*40)
    print(model.sympy())
    
    r2 = model.score(X_test, y_test)
    print(f"\n✅ Validation R^2 Score: {r2:.4f}")
    
    if r2 > 0.6:
        print("   (SUCCESS! We have reproduced the BATS scientific record.)")

if __name__ == "__main__":
    run_bats_discovery()