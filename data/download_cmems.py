"""Download CMEMS Multi-Observation Global Ocean Carbon data.

Dataset: Global Ocean Surface Carbon (Multi-Obs)
ID: cmems_obs-mob_glo_bgc-car_my_na_irr-i
Variables: spco2, ph, alkalinity, etc.
Resolution: 1° x 1° (native) or 0.25° (reprocessed)
Time: Monthly (1985-present)

Requirements:
    pip install copernicusmarine
"""

import sys
from pathlib import Path

try:
    import copernicusmarine
except ImportError:
    print("ERROR: copernicusmarine not installed. Run: pip install copernicusmarine")
    sys.exit(1)

def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_id = "cmems_obs-mob_glo_bgc-car_my_na_irr-i"
    
    print(f"Downloading {dataset_id}...")
    
    # Download Carbon (pCO2, pH)
    print(f"Downloading Carbon variables from {dataset_id}...")
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=["spco2", "ph", "talk"],
        start_datetime="2000-01-01T00:00:00",
        end_datetime="2023-12-31T23:59:59",
        minimum_longitude=-180,
        maximum_longitude=180,
        minimum_latitude=-90,
        maximum_latitude=90,
        output_directory=str(output_dir),
        output_filename="cmems_multiobs_carbon.nc",
        force_download=True,
    )

    # Download Physics (SST, SSS) from Multi-Obs Physics
    # ID: MULTIOBS_GLO_PHY_TS_SURFACE_MYNRT_015_001 (or similar)
    # common ID for reanalysis: cmems_obs-mob_glo_phy-ts_my_0.25deg_P1M-m
    phy_dataset_id = "cmems_obs-mob_glo_phy-ts_my_0.25deg_P1M-m"
    print(f"Downloading Physics variables from {phy_dataset_id}...")
    try:
        copernicusmarine.subset(
            dataset_id=phy_dataset_id,
            variables=["sst", "sss"],
            start_datetime="2000-01-01T00:00:00",
            end_datetime="2023-12-31T23:59:59",
            minimum_longitude=-180,
            maximum_longitude=180,
            minimum_latitude=-90,
            maximum_latitude=90,
            output_directory=str(output_dir),
            output_filename="cmems_multiobs_phys.nc",
            force_download=True,
        )
    except Exception as e:
        print(f"Warning: Failed to download Physics: {e}")
        print("You may need to check the dataset ID for Multi-Obs Physics.")

    print("Download complete.")

if __name__ == "__main__":
    main()
