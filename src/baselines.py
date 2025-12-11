import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

# Config
DATA_PATH = Path("data/03_processed/training_set.parquet")
LAT_MIN, LAT_MAX = 25, 35
LON_MIN, LON_MAX = -70, -60

def run_baselines():
    print("🔬 Loading Data for Baselines...")
    df = pd.read_parquet(DATA_PATH)
    
    # Filter BATS
    df = df[
        (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX) &
        (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
    ].copy()
    
    # Engineer Seasons (Same as AI)
    df['Season_Sin'] = np.sin(2 * np.pi * (df['Year'] % 1))
    df['Season_Cos'] = np.cos(2 * np.pi * (df['Year'] % 1))
    
    X = df[["SST", "Salinity", "Year", "Season_Sin", "Season_Cos"]]
    y = df["fCO2"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("-" * 30)
    print("📊 BASELINE RESULTS")
    print("-" * 30)
    
    # 1. Linear Regression (The "Dumb" Baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)
    print(f"1️⃣  Linear Regression R²: {lr_score:.4f}")
    
    # 2. Random Forest (The "Black Box" Ceiling)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    print(f"2️⃣  Random Forest R²:     {rf_score:.4f}")
    
    # 3. Compare with your PySR Result (Hardcoded from your run)
    print("3️⃣  PySR (Your AI) R²:    0.7684")
    
    print("-" * 30)
    print("INTERPRETATION:")
    if rf_score > 0.7684:
        print(f"PySR captures {(0.7684/rf_score)*100:.1f}% of the signal found by Random Forest,")
        print("but provides an explicit equation instead of a black box.")

if __name__ == "__main__":
    run_baselines()