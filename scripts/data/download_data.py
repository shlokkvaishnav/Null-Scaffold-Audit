"""Download SOCAT and Copernicus chlorophyll data with robust error handling.

Data Sources:
- SOCAT v2025: NOAA NCEI dataset 0304549 (gridded monthly fCO₂, SST, SSS)
- Copernicus Marine: Global biogeochemical reanalysis (chlorophyll-a)
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Load environment variables if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import copernicusmarine
except ImportError:
    copernicusmarine = None

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# SOCAT gridded monthly fCO₂ (NOAA NCEI OCADS 0304549)
SOCAT_URLS = [
    "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0304549/SOCATv2025_Gridded_Data/SOCATv2025_tracks_gridded_monthly.nc",
    "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0304549/SOCATv2024_Gridded_Data/SOCATv2024_tracks_gridded_monthly.nc",
]
SOCAT_FILENAME = "SOCATv2025_tracks_gridded_monthly.nc"

# Copernicus Marine Service (Global Ocean Biogeochemistry)
CMEMS_DATASET_ID = "cmems_mod_glo_bgc_my_0.25deg_P1M-m"
CMEMS_VARIABLE = "chl"
CMEMS_FILENAME = "cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc"

# Temporal coverage
START_DATE = "2015-01-01"  # Match config.py START_YEAR
END_DATE = "2024-12-31"    # Match config.py END_YEAR

# Download parameters
CHUNK_SIZE = 8192
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# =============================================================================
# AUTHENTICATION
# =============================================================================

def check_copernicus_credentials() -> bool:
    """Verify Copernicus Marine Service credentials.
    
    Returns:
        True if credentials found, False otherwise
    """
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")
    
    if not username or not password:
        logger.warning(
            "Copernicus credentials not found in environment.\n"
            "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in .env file.\n"
            "Or run: copernicusmarine login"
        )
        return False
    
    logger.info("✓ Copernicus credentials found")
    return True


# =============================================================================
# FILE VALIDATION
# =============================================================================

def compute_file_hash(path: Path, algorithm: str = "md5") -> str:
    """Compute hash of file for integrity checking."""
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_netcdf(path: Path) -> bool:
    """Check if NetCDF file is valid and non-empty.
    
    Args:
        path: Path to NetCDF file
        
    Returns:
        True if valid, False otherwise
    """
    if not path.exists():
        return False
    
    # Size check (NetCDF header + minimal data)
    min_size = 10_000  # 10 KB
    if path.stat().st_size < min_size:
        logger.warning(f"File too small (<{min_size} bytes): {path}")
        return False
    
    # Try opening with xarray
    try:
        import xarray as xr
        with xr.open_dataset(path, engine="netcdf4") as ds:
            # Check has variables
            if len(ds.data_vars) == 0:
                logger.warning(f"No data variables in {path}")
                return False
        return True
    except Exception as e:
        logger.warning(f"NetCDF validation failed for {path}: {e}")
        return False


# =============================================================================
# SOCAT DOWNLOAD
# =============================================================================

def download_file_with_retry(
    url: str,
    output_path: Path,
    max_retries: int = MAX_RETRIES,
    timeout: int = 120,
) -> bool:
    """Download file with progress bar and retry logic.
    
    Args:
        url: Download URL
        output_path: Destination path
        max_retries: Maximum retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading from {url} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(
                url, 
                stream=True, 
                timeout=timeout,
                headers={"User-Agent": "SD-MoSE/1.0"}
            )
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            
            # Download with progress bar
            with open(output_path, "wb") as f, \
                 tqdm(
                     desc=output_path.name,
                     total=total_size,
                     unit="B",
                     unit_scale=True,
                     unit_divisor=1024,
                 ) as progress_bar:
                
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            logger.info(f"✓ Downloaded: {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download failed (attempt {attempt + 1}): {e}")
            
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"All retry attempts exhausted for {url}")
                return False
    
    return False


def download_socat(output_dir: Path) -> bool:
    """Download SOCAT gridded monthly data.
    
    Args:
        output_dir: Directory to save file
        
    Returns:
        True if successful (already exists or downloaded), False otherwise
    """
    file_path = output_dir / SOCAT_FILENAME
    
    # Check if already downloaded and valid
    if file_path.exists():
        logger.info(f"SOCAT file exists: {file_path}")
        if validate_netcdf(file_path):
            logger.info("✓ SOCAT file validated")
            return True
        else:
            logger.warning("Existing SOCAT file invalid, re-downloading...")
            file_path.unlink()
    
    # Try URLs in order (v2025 → v2024)
    for url in SOCAT_URLS:
        version = url.split("/")[-2]  # e.g., "SOCATv2025_Gridded_Data"
        logger.info(f"Attempting {version}...")
        
        if download_file_with_retry(url, file_path):
            # Validate downloaded file
            if validate_netcdf(file_path):
                logger.info(f"✓ SOCAT download complete and validated")
                return True
            else:
                logger.warning("Downloaded file failed validation")
                file_path.unlink()
    
    logger.error("❌ SOCAT download failed for all URLs")
    return False


# =============================================================================
# COPERNICUS CHLOROPHYLL DOWNLOAD
# =============================================================================

def download_chlorophyll(output_dir: Path) -> bool:
    """Download Copernicus Marine Service chlorophyll data.
    
    Args:
        output_dir: Directory to save file
        
    Returns:
        True if successful, False otherwise
    """
    if copernicusmarine is None:
        logger.error(
            "❌ copernicusmarine package not installed.\n"
            "Install: pip install copernicusmarine"
        )
        return False
    
    file_path = output_dir / CMEMS_FILENAME
    
    # Check if already downloaded and valid
    if file_path.exists():
        logger.info(f"Chlorophyll file exists: {file_path}")
        if validate_netcdf(file_path):
            logger.info("✓ Chlorophyll file validated")
            return True
        else:
            logger.warning("Existing chlorophyll file invalid, re-downloading...")
            file_path.unlink()
    
    # Verify credentials
    if not check_copernicus_credentials():
        logger.error(
            "❌ Copernicus credentials missing.\n"
            "Solution 1: Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in .env\n"
            "Solution 2: Run 'copernicusmarine login' in terminal"
        )
        return False
    
    # Download via Copernicus Marine Toolbox
    logger.info(f"Downloading Copernicus chlorophyll ({START_DATE} to {END_DATE})...")
    logger.info("This may take 10-30 minutes depending on your connection...")
    
    try:
        copernicusmarine.subset(
            dataset_id=CMEMS_DATASET_ID,
            variables=[CMEMS_VARIABLE],
            start_datetime=START_DATE,
            end_datetime=END_DATE,
            minimum_longitude=-180.0,
            maximum_longitude=180.0,
            minimum_latitude=-90.0,
            maximum_latitude=90.0,
            minimum_depth=0.0,
            maximum_depth=10.0,  # Surface layer only
            output_directory=str(output_dir),
            output_filename=CMEMS_FILENAME,
            force_download=True,
        )
        
        # Validate
        if validate_netcdf(file_path):
            logger.info(f"✓ Chlorophyll download complete and validated")
            return True
        else:
            logger.error("Downloaded chlorophyll file failed validation")
            return False
            
    except Exception as e:
        logger.error(f"❌ Copernicus download failed: {e}")
        logger.info(
            "Troubleshooting:\n"
            "1. Verify credentials: copernicusmarine login\n"
            "2. Check dataset availability at https://data.marine.copernicus.eu/\n"
            "3. Ensure network connection is stable"
        )
        return False


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Download all required datasets."""
    # Add src to path for config import
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    
    try:
        from climate_discovery.config import RAW_DIR
    except ImportError:
        logger.error("Cannot import config. Ensure src/climate_discovery exists.")
        sys.exit(1)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Download SOCAT and Copernicus chlorophyll data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py                    # Use default RAW_DIR
  python download_data.py --data_dir /path   # Custom output directory
  
Environment variables (.env file):
  COPERNICUS_USERNAME=your_username
  COPERNICUS_PASSWORD=your_password
        """
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=f"Output directory (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--skip-socat",
        action="store_true",
        help="Skip SOCAT download (for testing)",
    )
    parser.add_argument(
        "--skip-chlorophyll",
        action="store_true",
        help="Skip chlorophyll download (for testing)",
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = Path(args.data_dir) if args.data_dir else RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Download datasets
    success = True
    
    if not args.skip_socat:
        logger.info("=" * 60)
        logger.info("DOWNLOADING SOCAT DATA")
        logger.info("=" * 60)
        if not download_socat(output_dir):
            logger.error("❌ SOCAT download failed")
            success = False
    
    if not args.skip_chlorophyll:
        logger.info("=" * 60)
        logger.info("DOWNLOADING COPERNICUS CHLOROPHYLL")
        logger.info("=" * 60)
        if not download_chlorophyll(output_dir):
            logger.error("❌ Chlorophyll download failed")
            success = False
    
    # Summary
    logger.info("=" * 60)
    if success:
        logger.info("✓ All downloads completed successfully!")
        logger.info(f"Files saved to: {output_dir}")
        logger.info("\nNext step: python -m scripts.data.preprocess_data")
    else:
        logger.error("❌ Some downloads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()