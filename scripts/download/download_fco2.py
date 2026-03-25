"""
Download fCO2 data from SOCAT Gridded Product v2023

Downloads gridded surface ocean fCO2 monthly means.
"""

import sys
from pathlib import Path
from datetime import datetime
import hashlib
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.schema import DataContract


def download_socat(output_dir: Path) -> Path:
    """
    Download SOCAT gridded fCO2 product.
    
    Returns:
        Path to downloaded file
    """
    # SOCAT v2023 gridded monthly from NCEI (3.7 GB)
    url = "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0278913/SOCATv2023_Gridded_Dat/SOCATv2023_tracks_gridded_monthly.nc"
    filename = "SOCATv2023_tracks_gridded_monthly.nc"
    
    output_file = output_dir / filename
    
    print(f"Downloading SOCAT v2023 Gridded Monthly fCO2")
    print(f"URL: {url}")
    print(f"Saving to: {output_file}\n")
    
    try:
        import requests
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"File size: {total_size / 1e9:.2f} GB")
        
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
        print(f"\nFailed to download {url}: {e}")
        print("\nAlternative:")
        print("1. Visit https://www.socat.info/index.php/version-2023/")
        print("2. Download: SOCATv2023_tracks_gridded_monthly.nc")
        print(f"3. Save to {output_dir}/")
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
        "source": DataContract.SOURCES["fco2"],
        "download_date": datetime.now().isoformat(),
        "url": "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0278913/SOCATv2023_Gridded_Dat/SOCATv2023_tracks_gridded_monthly.nc",
        "file": str(filepath.name) if filepath else None,
        "checksum": checksum,
        "time_range": "1957-2023 (monthly 1° grid)",
        "notes": "Single gridded product from NCEI, subset to 2010-2020 in preprocessing"
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_file}")


def main():
    """Download SOCAT gridded fCO2 data."""
    output_dir = Path("data/raw/fco2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DOWNLOADING fCO2 DATA (SOCAT v2023)")
    print("="*60 + "\n")
    
    # Check if file already exists
    output_file = output_dir / "SOCATv2023_tracks_gridded_monthly.nc"
    
    if output_file.exists():
        print(f"File already exists: {output_file}")
        filepath = output_file
    else:
        filepath = download_socat(output_dir)
    
    if filepath and filepath.exists():
        checksum = compute_checksum(filepath)
        print(f"MD5: {checksum}")
        save_metadata(output_dir, filepath, checksum)
        
        print(f"\n{'='*60}")
        print(f"File size: {filepath.stat().st_size / 1e9:.2f} GB")
        print(f"{'='*60}")
        print("\nfCO2 download complete")
        print("  Time range: 1957-2023")
        print("  Resolution: 1° monthly")
        print("  Next: Preprocessing will subset to 2010-2020")
    else:
        print("\nDownload failed - see alternative instructions above")


if __name__ == "__main__":
    main()
