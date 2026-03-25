"""
Download Chlorophyll-a data from NASA MODIS-Aqua Level 3 Mapped Monthly

Downloads monthly 4km chlorophyll-a composites.
"""

import sys
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.schema import DataContract


def get_earthdata_auth():
    """
    Get NASA Earthdata credentials from .netrc or environment variables.
    
    Returns:
        Tuple of (username, password) or None if not found
    """
    # Try .netrc file first (recommended method)
    try:
        from netrc import netrc
        n = netrc()
        auth = n.authenticators("urs.earthdata.nasa.gov")
        if auth:
            return (auth[0], auth[2])  # (username, password)
    except Exception:
        pass
    
    # Fallback to environment variables
    username = os.getenv('EARTHDATA_USERNAME')
    password = os.getenv('EARTHDATA_PASSWORD')
    if username and password:
        return (username, password)
    
    return None


def is_valid_netcdf(filepath: Path) -> bool:
    """
    Check if file is a valid NetCDF file (not an HTML error page).
    
    Args:
        filepath: Path to file to check
        
    Returns:
        True if valid NetCDF, False otherwise
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(20)
            # Check for NetCDF/HDF5 magic numbers
            if header[:3] in [b'CDF', b'\x89HD'] or header[:4] == b'\x89HDF':
                return True
            # Check for HTML error pages
            if b'<!DOCTYPE' in header or b'<html' in header.lower():
                return False
        return False
    except Exception:
        return False


def download_modis_month(year: int, month: int, output_dir: Path) -> Path:
    """
    Download MODIS-Aqua chlorophyll for a specific month.
    
    Args:
        year: Year to download
        month: Month to download (1-12)
        output_dir: Directory to save file
        
    Returns:
        Path to downloaded file
    """
    # NASA OceanColor direct data access
    base_url = "https://oceandata.sci.gsfc.nasa.gov/ob/getfile"
    
    # Filename pattern for monthly L3 mapped CHL product
    # Example: AQUA_MODIS.20100101_20100131.L3m.MO.CHL.chlor_a.4km.nc
    # Construct proper date range
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    filename = f"AQUA_MODIS.{year}{month:02d}01_{year}{month:02d}{last_day}.L3m.MO.CHL.chlor_a.4km.nc"
    url = f"{base_url}/{filename}"
    
    output_file = output_dir / filename
    
    # Check if file exists and is valid
    if output_file.exists():
        if is_valid_netcdf(output_file):
            print(f"  ✓ Already exists: {filename}")
            return output_file
        else:
            print(f"  ⚠ Cleaning corrupted file: {filename}")
            output_file.unlink()  # Delete corrupted file
    
    print(f"  Downloading: {filename}")
    
    # Get authentication
    auth = get_earthdata_auth()
    if not auth:
        print(f"    ⚠ Warning: No Earthdata credentials found")
        print(f"    Create .netrc file or set EARTHDATA_USERNAME/EARTHDATA_PASSWORD")
    
    try:
        import requests
        
        # Add authentication and required headers
        headers = {
            'User-Agent': 'climate-equation-discovery/1.0'
        }
        
        response = requests.get(
            url, 
            auth=auth,
            headers=headers,
            stream=True, 
            timeout=180,
            allow_redirects=True
        )
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Validate downloaded file
        if not is_valid_netcdf(output_file):
            print(f"    ✗ Downloaded file is not valid NetCDF (likely HTML error page)")
            # Show first few bytes for debugging
            with open(output_file, 'rb') as f:
                first_bytes = f.read(100)
                if b'<html' in first_bytes.lower():
                    print(f"    File contains HTML. Authentication may have failed.")
            output_file.unlink()  # Clean up invalid file
            return None
        
        print(f"    ✓ Downloaded and validated")
        return output_file
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        if output_file.exists():
            output_file.unlink()  # Clean up partial download
        return None


def compute_checksum(filepath: Path) -> str:
    """Compute MD5 checksum of file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_metadata(output_dir: Path, files: list, checksums: dict) -> None:
    """Save download metadata."""
    metadata = {
        "source": DataContract.SOURCES["chl"],
        "download_date": datetime.now().isoformat(),
        "time_range": {
            "start": DataContract.TEMPORAL["start_date"],
            "end": DataContract.TEMPORAL["end_date"],
        },
        "files": [str(f.name) for f in files] if files else [],
        "checksums": checksums,
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved to {metadata_file}")


def main():
    """Download all Chl-a data for contract period."""
    output_dir = Path("data/raw/chl")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_date = datetime.fromisoformat(DataContract.TEMPORAL["start_date"])
    end_date = datetime.fromisoformat(DataContract.TEMPORAL["end_date"])
    
    print("="*60)
    print("DOWNLOADING CHLOROPHYLL-A DATA (MODIS-Aqua)")
    print("="*60)
    print(f"\nPeriod: {start_date.date()} to {end_date.date()}\n")
    
    downloaded_files = []
    checksums = {}
    
    year = start_date.year
    month = start_date.month
    
    while (year < end_date.year) or (year == end_date.year and month <= end_date.month):
        filepath = download_modis_month(year, month, output_dir)
        
        if filepath and filepath.exists():
            downloaded_files.append(filepath)
            checksum = compute_checksum(filepath)
            checksums[filepath.name] = checksum
        
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    save_metadata(output_dir, downloaded_files, checksums)
    
    print(f"\n{'='*60}")
    print(f"Downloaded {len(downloaded_files)} files")
    if downloaded_files:
        print(f"Total size: {sum(f.stat().st_size for f in downloaded_files) / 1e9:.2f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
