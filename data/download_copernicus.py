"""Download Copernicus CMEMS chlorophyll data.

Copernicus Marine Service provides global ocean biogeochemistry products.
This script downloads chlorophyll-a concentration (2000-2023).

Product: GLOBAL_MULTIYEAR_BGC_001_029
Variable: Chlorophyll-a mass concentration (chl)
Resolution: 0.25° x 0.25°
Time range: 2000-2023

REQUIREMENTS:
1. Create free account: https://data.marine.copernicus.eu/register
2. Install: pip install copernicusmarine
3. Configure credentials (prompted on first run)
"""

import sys
from pathlib import Path

try:
    import copernicusmarine
except ImportError:
    print("✗ ERROR: copernicusmarine package not installed")
    print()
    print("Install with:")
    print("  pip install copernicusmarine")
    print()
    sys.exit(1)


def main():
    """Download Copernicus chlorophyll dataset."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = raw_dir / "cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc"
    
    if output_file.exists():
        print(f"✓ CMEMS file already exists: {output_file}")
        print("  To re-download, delete the file first.")
        return
    
    # Dataset parameters
    dataset_id = "cmems_mod_glo_bgc_my_0.25deg_P1M-m"
    variable = "chl"
    
    print("=" * 70)
    print("COPERNICUS CMEMS CHLOROPHYLL DOWNLOAD")
    print("=" * 70)
    print(f"Product: {dataset_id}")
    print(f"Variable: {variable} (Chlorophyll-a)")
    print(f"Time range: 2000-01-01 to 2023-12-31")
    print(f"Resolution: 0.25°")
    print(f"Expected size: ~2 GB")
    print()
    print("IMPORTANT:")
    print("1. You need a FREE Copernicus account")
    print("2. Register at: https://data.marine.copernicus.eu/register")
    print("3. You'll be prompted for credentials if not configured")
    print("=" * 70)
    print()
    
    try:
        print("Downloading... (this may take 15-30 minutes)")
        print()
        
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=[variable],
            start_datetime="2000-01-01T00:00:00",
            end_datetime="2023-12-31T23:59:59",
            minimum_longitude=-180,
            maximum_longitude=180,
            minimum_latitude=-90,
            maximum_latitude=90,
            output_filename=str(output_file),
            output_directory=str(raw_dir),
            force_download=True,
        )
        
        print()
        print(f"✓ Download complete: {output_file}")
        print()
        
        # Display file info
        size_mb = output_file.stat().st_size / 1e6
        print("File Information:")
        print(f"  - Path: {output_file}")
        print(f"  - Size: {size_mb:.1f} MB")
        print()
        print("Next step: Run data/preprocess.py to fuse SOCAT + CMEMS")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check your Copernicus credentials")
        print("2. Configure with: copernicusmarine login")
        print("3. Verify you have access to the dataset")
        print("4. Check your internet connection")
        print()
        print("Alternative: Download manually from:")
        print("https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_BGC_001_029")
        sys.exit(1)


if __name__ == "__main__":
    main()
