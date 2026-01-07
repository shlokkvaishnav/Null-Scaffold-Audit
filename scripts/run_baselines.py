import xarray as xr
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import logging
from pathlib import Path

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIG ---
DATA_PATH = Path("data/03_processed/climate_fused_dataset.nc")
TEST_YEAR_START = 2020  # Hold out 2020-2024 for testing
FEATURES = ['sst', 'sss', 'sin_month', 'cos_month', 'log_chl','year']
TARGET = 'fco2'

def load_and_flatten_data():
    if not DATA_PATH.exists():
        logger.error(f"❌ Data not found at {DATA_PATH}.")
        return None, None, None, None

    logger.info("loading dataset...")
    ds = xr.open_dataset(DATA_PATH)
    
    # Convert 3D grid to Table (Lat/Lon/Time -> Rows)
    df = ds.to_dataframe().reset_index()
    
    # Drop rows with missing data (Land/Clouds)
    initial_len = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    logger.info(f"   Dropped {initial_len - len(df)} rows (NaNs). Final: {len(df)}")

    # Time Split (Train vs Test)
    train_mask = df['year'] < TEST_YEAR_START
    test_mask = df['year'] >= TEST_YEAR_START

    X_train = df.loc[train_mask, FEATURES]
    y_train = df.loc[train_mask, TARGET]
    X_test = df.loc[test_mask, FEATURES]
    y_test = df.loc[test_mask, TARGET]
    
    logger.info(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, y_train, X_test, y_test

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n📊 --- {name} RESULTS ---")
    print(f"   RMSE: {rmse:.4f} µatm")
    print(f"   R²:   {r2:.4f}")

def main():
    X_train, y_train, X_test, y_test = load_and_flatten_data()
    if X_train is None: 
        return

    # 1. Linear Baseline
    logger.info("🚀 Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    evaluate("Linear Regression", lr, X_test, y_test)
    
    # Check Coefficients
    print("\n   Linear Coefficients:")
    for f, c in zip(FEATURES, lr.coef_):
        print(f"     {f}: {c:.4f}")

    # 2. Random Forest (The Ceiling)
    logger.info("🌲 Training Random Forest (Upper Bound)...")
    # Limited depth for speed
    rf = RandomForestRegressor(n_estimators=30, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test)

if __name__ == "__main__":
    main()