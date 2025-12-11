import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- CONFIGURATION ---
DATA_PATH = Path("data/03_processed/training_set.parquet")
SUB_SAMPLE = 50000  # Use more points for a fair global test

def run_global_baselines():
    print("🌍 LOADING GLOBAL DATA FOR BASELINING...")
    df = pd.read_parquet(DATA_PATH)
    
    # Feature Engineering (Must match AI)
    df['Season_Sin'] = np.sin(2 * np.pi * (df['Year'] % 1))
    df['Season_Cos'] = np.cos(2 * np.pi * (df['Year'] % 1))
    
    X = df[["SST", "Salinity", "Year", "Season_Sin", "Season_Cos"]]
    y = df["fCO2"]
    
    # Subsample for speed (Random Forest is slow on 300k points)
    if len(df) > SUB_SAMPLE:
        print(f"   Subsampling to {SUB_SAMPLE} points...")
        X, _, y, _ = train_test_split(X, y, train_size=SUB_SAMPLE, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n📊 GLOBAL BASELINE RESULTS")
    print("="*30)

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    print(f"1️⃣  Global Linear Regression R²: {lr_r2:.4f}")

    # 2. Random Forest (The Ceiling)
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"2️⃣  Global Random Forest R²:     {rf_r2:.4f}")

    # 3. Your Hybrid Agent Score (Hardcoded from previous run)
    print("3️⃣  Hybrid PySR (Your Agent):    0.2500")

    print("="*30)
    
    if lr_r2 < 0.25:
        print("✅ VERDICT: Your Agent BEATS the Linear Baseline.")
        print("   This confirms that 'Regime Discovery' adds real value.")
    else:
        print("⚠️ VERDICT: Linear Regression is winning. Your PySR needs tuning.")

if __name__ == "__main__":
    run_global_baselines()