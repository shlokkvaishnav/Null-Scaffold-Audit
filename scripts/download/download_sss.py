"""
Download SSS data from EN4.2.2 (Met Office Hadley Centre)

Downloads yearly ZIP files containing monthly sea surface salinity.
No authentication required.
"""

import sys
from pathlib import Path
from datetime import datetime
import hashlib
import json
import zipfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.schema import DataContract


def download_en4_year(year: int, output_dir: Path) -> Path:
    """
    Download EN4 salinity data for an entire year (ZIP file with 12 monthly files).
    
    Args:
        year: Year to download
        output_dir: Directory to save file
        
    Returns:
        Path to downloaded ZIP file
    """
    # Met Office EN4.2.2 analyses (yearly ZIP files)
    base_url = "https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1/EN.4.2.2"
    
    # Filename pattern: EN.4.2.2.analyses.g10.YYYY.zip
    filename = f"EN.4.2.2.analyses.g10.{year}.zip"
    url = f"{base_url}/{filename}"
    
    output_file = output_dir / filename
    
    if output_file.exists():
        print(f"  Already exists: {filename}")
        return output_file
    
    print(f"  Downloading: {filename}")
    
    try:
        import requests
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r    Progress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n    Downloaded ({output_file.stat().st_size / 1e6:.1f} MB)")
        return output_file
        
    except Exception as e:
        print(f"\n    Failed: {e}")
        return None


def extract_zip(zip_path: Path, output_dir: Path) -> list:
    """
    Extract NetCDF files from ZIP archive.
    
    Args:
        zip_path: Path to ZIP file
        output_dir: Directory to extract to
        
    Returns:
        List of extracted file paths
    """
    print(f"  Extracting: {zip_path.name}")
    
    extracted_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract only .nc files
        for file_info in zip_ref.filelist:
            if file_info.filename.endswith('.nc'):
                zip_ref.extract(file_info, output_dir)
                extracted_file = output_dir / file_info.filename
                extracted_files.append(extracted_file)
    
    print(f"    Extracted {len(extracted_files)} NetCDF files")
    return extracted_files


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
        "source": {
            "name": "EN4.2.2 Objective Analysis",
            "url": "https://www.metoffice.gov.uk/hadobs/en4/",
            "variable": "salinity",
            "units": "PSU",
            "resolution": "1° monthly (g10 grid)"
        },
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
    
    print(f"\nMetadata saved to {metadata_file}")


def main():
    """Download EN4 SSS data."""
    output_dir = Path("data/raw/sss")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_date = datetime.fromisoformat(DataContract.TEMPORAL["start_date"])
    end_date = datetime.fromisoformat(DataContract.TEMPORAL["end_date"])
    
    print("="*60)
    print("DOWNLOADING SSS DATA (EN4.2.2)")
    print("="*60)
    print(f"\nPeriod: {start_date.date()} to {end_date.date()}")
    print("Source: Met Office Hadley Centre")
    print("Format: Yearly ZIP files (12 monthly NetCDF files each)")
    print("No authentication required\n")
    
    extracted_files = []
    checksums = {}
    
    # Download yearly ZIP files
    for year in range(start_date.year, end_date.year + 1):
        zip_file = download_en4_year(year, output_dir)
        
        if zip_file and zip_file.exists():
            # Extract NetCDF files from ZIP
            files = extract_zip(zip_file, output_dir)
            extracted_files.extend(files)
            
            # Compute checksums for extracted files
            for f in files:
                checksums[f.name] = compute_checksum(f)
            
            # Delete ZIP file to save space
            zip_file.unlink()
            print(f"    Deleted ZIP file to save space\n")
    
    save_metadata(output_dir, extracted_files, checksums)
    
    print(f"\n{'='*60}")
    print(f"Downloaded and extracted {len(extracted_files)} monthly files")
    if extracted_files:
        print(f"Total size: {sum(f.stat().st_size for f in extracted_files) / 1e9:.2f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
