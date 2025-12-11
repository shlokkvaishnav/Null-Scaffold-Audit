import xarray as xr
import numpy as np
from pathlib import Path

# --- Configuration ---
RAW_DATA_PATH = Path("data/01_raw/SOCATv2025_tracks_gridded_monthly.nc")
PROCESSED_DATA_PATH = Path("data/03_processed/training_set.parquet")

VARS = {
    "fco2_ave_unwtd": "fCO2",
    "sst_ave_unwtd": "SST",
    "salinity_ave_unwtd": "Salinity"
}

def process_data():
    print("⏳ Loading NetCDF...")
    ds = xr.open_dataset(RAW_DATA_PATH, chunks={"tmnth": 10})
    
    # 1. Rename dims
    ds = ds.rename({"tmnth": "time", "ylat": "lat", "xlon": "lon"})
    
    # 2. Select variables
    subset = ds[list(VARS.keys())]
    subset = subset.rename(VARS)
    
    # 3. Flatten to DataFrame
    print("🚜 Flattening to table...")
    df = subset.to_dataframe().reset_index()
    
    # --- FEATURE ENGINEERING (The Fix) ---
    print("Feature Engineering: Adding Time & Space...")
    
    # Time: Decimal Year (for the trend)
    df['Year'] = df['time'].dt.year + df['time'].dt.dayofyear / 365.25
    
    # Space: Absolute Latitude (Equator=0, Poles=90)
    # Physics is symmetric: 40°N behaves similarly to 40°S
    df['AbsLat'] = np.abs(df['lat'])
    
    # 4. Cleaning
    print("🧹 Cleaning missing values...")
    clean_df = df.dropna(subset=["fCO2", "SST", "Salinity", "Year", "AbsLat"])
    
    # 5. Physics Guardrails (Stricter)
    clean_df = clean_df[
        (clean_df["SST"] > -2) & (clean_df["SST"] < 35) &
        (clean_df["Salinity"] > 20) & (clean_df["Salinity"] < 40)
    ]
    
    print(f"✅ CLEAN DATA POINTS: {len(clean_df):,}")
    
    # 6. Save
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(PROCESSED_DATA_PATH)
    print(f"💾 Saved for AI: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    process_data()