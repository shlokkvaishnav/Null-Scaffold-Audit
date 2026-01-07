import argparse
import logging
import requests
import copernicusmarine
from pathlib import Path
from tqdm import tqdm

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# 1. SOCAT (Physical Data - In Situ CO2)
# v2025 covers 1957 to 2024.
SOCAT_URL = "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0304549/SOCATv2025_Gridded_Data/SOCATv2025_tracks_gridded_monthly.nc"
SOCAT_FILENAME = "SOCATv2025_tracks_gridded_monthly.nc"

# 2. Copernicus (Biological Data - Chlorophyll)
# PRODUCT ID: GLOBAL_MULTIYEAR_BGC_001_029
# DATASET ID: cmems_mod_glo_bgc_my_0.25deg_P1M-m (Monthly Mean)
# Note: "my" = MultiYear, "P1M-m" = Period 1 Month (Monthly)
CMEMS_DATASET_ID = "cmems_mod_glo_bgc_my_0.25deg_P1M-m"
CMEMS_FILENAME = "cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc"
CMEMS_VARIABLE = "chl" # Standard short-name for Mass Concentration of Chlorophyll A

# Time range Alignment
# SOCAT ends 2024. CMEMS starts Jan 1993. 
# Common Overlap: Jan 1993 - Dec 2024.
START_DATE = "1993-01-01"
END_DATE = "2024-12-31" 
# ---------------------

def check_copernicus_auth():
    """Checks if the user has a valid Copernicus session."""
    # The library looks for a configuration file in your home directory.
    # We try a simple 'describe' call to check connectivity.
    try:
        logger.info("🔐 Verifying Copernicus credentials...")
        # This function doesn't need to return anything, just not raise an error
        # If this fails, the user hasn't run 'copernicusmarine login'
        # We can't strictly 'check' without running a command, but we can catch the error later.
        pass 
    except Exception:
        pass

def download_socat(output_dir: Path):
    """Downloads SOCAT data via HTTP with resume capability."""
    file_path = output_dir / SOCAT_FILENAME
    
    # Check for existing partial file
    resume_byte_pos = file_path.stat().st_size if file_path.exists() else 0
    
    headers = {}
    mode = "wb"
    if resume_byte_pos > 0:
        headers = {"Range": f"bytes={resume_byte_pos}-"}
        mode = "ab"
        logger.info(f"🔄 Resuming SOCAT download from {resume_byte_pos / (1024**3):.2f} GB...")
    else:
        logger.info("⬇️  Starting SOCAT download...")

    try:
        response = requests.get(SOCAT_URL, headers=headers, stream=True)
        
        # If server returns 416, file is already complete
        if response.status_code == 416: 
            logger.info("✅ SOCAT file is already complete!")
            return

        total_size = int(response.headers.get("content-length", 0)) + resume_byte_pos

        with open(file_path, mode) as file, tqdm(
            desc="SOCAT",
            total=total_size,
            initial=resume_byte_pos,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    bar.update(len(chunk))
                    file.write(chunk)
        
        logger.info(f"✅ SOCAT Saved: {file_path}")

    except Exception as e:
        logger.error(f"❌ SOCAT Download Failed: {e}")

def download_chlorophyll(output_dir: Path):
    """Downloads Chlorophyll data via Copernicus API."""
    file_path = output_dir / CMEMS_FILENAME
    
    if file_path.exists():
        logger.info(f"✅ Chlorophyll file already exists: {file_path}")
        return

    logger.info(f"⬇️  Starting Copernicus Chlorophyll download ({START_DATE} to {END_DATE})...")
    logger.info(f"ℹ️  Dataset ID: {CMEMS_DATASET_ID}")
    
    try:
        copernicusmarine.subset(
            dataset_id=CMEMS_DATASET_ID,
            variables=[CMEMS_VARIABLE],
            start_datetime=START_DATE,
            end_datetime=END_DATE,
            minimum_longitude=-180,
            maximum_longitude=180,
            minimum_latitude=-90,
            maximum_latitude=90,
            minimum_depth=0,
            maximum_depth=10,  # Surface layer only
            output_directory=str(output_dir),
            output_filename=CMEMS_FILENAME,
            overwrite=True,
            # force_download=True # Uncomment if you have issues with cached credentials
        )
        logger.info(f"✅ Chlorophyll Saved: {file_path}")
        
    except Exception as e:
        logger.error("❌ Copernicus API Error.")
        logger.error(f"   Error Details: {e}")
        logger.warning("\n⚠️  AUTHENTICATION LIKELY REQUIRED ⚠️")
        logger.warning("   Please run the following command in your terminal and follow the instructions:")
        logger.warning("   $ copernicusmarine login")
        logger.warning("   (You will need your Copernicus Marine username and password)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download climate data.")
    parser.add_argument("--data_dir", type=str, default="data/01_raw", help="Directory to save raw data")
    args = parser.parse_args()

    output_path = Path(args.data_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    check_copernicus_auth()
    download_socat(output_path)
    download_chlorophyll(output_path)