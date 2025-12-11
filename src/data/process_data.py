import xarray as xr
from pathlib import Path

#Configuration
RAW_DATA_PATH = Path("data/01_raw/SOCATv2025_tracks_gridded_monthly.nc")
PROCESSED_DATA_PATH = Path("data/03_processed/training_set.parquet")

# Exact mapping based on your notebook analysis
VARS = {
    "fco2_ave_unwtd": "fCO2",    # Target Variable
    "sst_ave_unwtd": "SST",      # Temperature
    "salinity_ave_unwtd": "Salinity" # Salt content
}

def process_data():
    print("⏳ Loading NetCDF (Lazy)...")
    # Open with chunks to handle 4GB size
    ds = xr.open_dataset(RAW_DATA_PATH, chunks={"tmnth": 10})
    
    # 1. Standardize Dimensions (tmnth -> time)
    # This ensures your final table has readable columns like 'time', 'lat', 'lon'
    ds = ds.rename({"tmnth": "time", "ylat": "lat", "xlon": "lon"})
    
    # 2. Select only the physics variables
    subset = ds[list(VARS.keys())]
    
    # 3. Rename to friendly names (e.g., 'sst_ave_unwtd' -> 'SST')
    subset = subset.rename(VARS)
    
    # 4. Flatten to Table (The "Cube to Sheet" conversion)
    print("🚜 Flattening to table (this determines the dataset size)...")
    df = subset.to_dataframe().reset_index()
    
    # 5. Drop Missing Data
    # We can't train on rows where we have CO2 but no Temperature.
    print("🧹 Cleaning missing values...")
    clean_df = df.dropna(subset=["fCO2", "SST", "Salinity"])
    
    # 6. Filter Outliers (Basic Physics Guardrails)
    # Remove obvious sensor errors (e.g., SST < -2 degrees or Salinity < 0)
    clean_df = clean_df[
        (clean_df["SST"] > -5) & (clean_df["SST"] < 40) &
        (clean_df["Salinity"] > 0) & (clean_df["Salinity"] < 50)
    ]
    
    count = len(clean_df)
    print(f"✅ CLEAN DATA POINTS: {count:,}")
    
    # 7. Save to Parquet
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(PROCESSED_DATA_PATH)
    print(f"💾 Saved for AI: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    process_data()