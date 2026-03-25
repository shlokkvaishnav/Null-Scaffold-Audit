"""
Download SST data from NOAA PSL OISST v2.1 Monthly Mean

Downloads single consolidated file with monthly means (1981-present).
"""

import sys
from pathlib import Path
from datetime import datetime
import hashlib
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.schema import DataContract


def download_sst_monthly(output_dir: Path) -> Path:
    """
    Download NOAA OISST monthly mean file.
    
    This is a single ~500MB file containing the full time series.
    
    Returns:
        Path to downloaded file
    """
    # NOAA PSL consolidated monthly mean file
    url = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
    filename = "sst.mon.mean.nc"
    output_file = output_dir / filename
    
    print(f"Downloading NOAA OISST Monthly Mean")
    print(f"URL: {url}")
    print(f"Saving to: {output_file}\n")
    
    try:
        import requests
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"File size: {total_size / 1e6:.1f} MB")
        
        with open(output_file, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print(f"\nDownloaded: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"\nFailed to download: {e}")
        return None


def compute_checksum(filepath: Path) -> str:
    """Compute MD5 checksum of file."""
    print("\nComputing checksum...")
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_metadata(output_dir: Path, filepath: Path, checksum: str) -> None:
    """Save download metadata."""
    metadata = {
        "source": DataContract.SOURCES["sst"],
        "download_date": datetime.now().isoformat(),
        "url": "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc",
        "file": str(filepath.name),
        "checksum": checksum,
        "time_range": "1981-present (monthly means)",
        "notes": "Single consolidated file, subset to 2010-2020 in preprocessing"
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_file}")


def main():
    """Download NOAA OISST monthly mean SST data."""
    output_dir = Path("data/raw/sst")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DOWNLOADING SST DATA")
    print("="*60 + "\n")
    
    # Check if file already exists
    output_file = output_dir / "sst.mon.mean.nc"
    
    if output_file.exists():
        print(f"File already exists: {output_file}")
        print("Verifying checksum...")
        filepath = output_file
    else:
        filepath = download_sst_monthly(output_dir)
    
    if filepath and filepath.exists():
        checksum = compute_checksum(filepath)
        print(f"MD5: {checksum}")
        save_metadata(output_dir, filepath, checksum)
        
        print(f"\n{'='*60}")
        print(f"File size: {filepath.stat().st_size / 1e6:.1f} MB")
        print(f"{'='*60}")
        print("\nSST download complete")
        print("  Time range: 1981-present")
        print("  Resolution: 0.25° daily → monthly means")
        print("  Next: Preprocessing will subset to 2010-2020")
    else:
        print("\nDownload failed")


if __name__ == "__main__":
    main()
