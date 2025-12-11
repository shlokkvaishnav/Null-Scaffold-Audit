import yaml
import pandas as pd
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- Configuration ---
DATA_PATH = Path("data/03_processed/training_set.parquet")
CONFIG_PATH = Path("configs/pysr_config.yaml")
SAMPLE_SIZE = 25000 

def run_discovery():
    print("🔬 Loading Training Data...")
    df = pd.read_parquet(DATA_PATH)
    
    # 1. Subsample
    if len(df) > SAMPLE_SIZE:
        df_train_subset = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        df_train_subset = df

    # --- UPDATED INPUTS ---
    # Now we include 'AbsLat' (Location)
    X = df_train_subset[["SST", "Salinity", "Year", "AbsLat"]] 
    y = df_train_subset["fCO2"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    print("\n🧠 Initializing PySR...")
    model = PySRRegressor(**config)

    print("🚀 Starting Discovery Loop (Attempt 3: With Geography)...")
    model.fit(X_train, y_train)

    print("\n" + "="*40)
    print("🏆 THE DISCOVERED EQUATION")
    print("="*40)
    print(model.sympy())
    
    r2 = model.score(X_test, y_test)
    print(f"\n✅ Validation R^2 Score: {r2:.4f}")

if __name__ == "__main__":
    run_discovery()