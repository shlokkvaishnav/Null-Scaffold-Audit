"""Download SOCAT gridded dataset.

SOCAT (Surface Ocean CO₂ Atlas) provides quality-controlled surface ocean fCO₂ data.
This script downloads the gridded monthly product (v2025).

Data source: https://www.socat.info/
Product: Gridded monthly 1°x1° resolution
Variables: fCO₂, SST, SSS, latitude, longitude, time
Time range: 1957-2023
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

def download_progress(block_num, block_size, total_size):
    """Display download progress bar."""
    downloaded = block_num * block_size
    percent = min(100.0 * downloaded / total_size, 100)
    bar_length = 50
    filled = int(bar_length * percent / 100)
    bar = '█' * filled + '-' * (bar_length - filled)
    
    print(f'\r[{bar}] {percent:.1f}% ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)', end='')
    
    if downloaded >= total_size:
        print()  # New line when complete


def main():
    """Download SOCAT gridded dataset."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = raw_dir / "SOCATv2025_tracks_gridded_monthly.nc"
    
    if output_file.exists():
        print(f"✓ SOCAT file already exists: {output_file}")
        print("  To re-download, delete the file first.")
        return
    
    # SOCAT gridded dataset URL
    # Note: This is the 2024 version. Check https://www.socat.info/ for latest.
    socat_url = "https://www.socat.info/socat_files/v2024/SOCATv2024_tracks_gridded_monthly.nc"
    
    print("=" * 70)
    print("SOCAT GRIDDED DATASET DOWNLOAD")
    print("=" * 70)
    print(f"Source: {socat_url}")
    print(f"Destination: {output_file}")
    print(f"Size: ~500 MB")
    print()
    print("Note: This may take 5-10 minutes depending on your connection.")
    print("=" * 70)
    print()
    
    try:
        print("Downloading...")
        urlretrieve(socat_url, output_file, reporthook=download_progress)
        print()
        print(f"✓ Download complete: {output_file}")
        print()
        
        # Display file info
        size_mb = output_file.stat().st_size / 1e6
        print("File Information:")
        print(f"  - Path: {output_file}")
        print(f"  - Size: {size_mb:.1f} MB")
        print()
        print("Next step: Run data/download_copernicus.py")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the URL is still valid at https://www.socat.info/")
        print("3. Try downloading manually and place in data/raw/")
        sys.exit(1)


if __name__ == "__main__":
    main()
