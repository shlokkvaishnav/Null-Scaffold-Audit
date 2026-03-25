"""
Verify checksums for all downloaded data files.
"""

import json
from pathlib import Path
import hashlib


def compute_checksum(filepath: Path) -> str:
    """Compute MD5 checksum of file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_directory(data_dir: Path) -> dict:
    """
    Verify all files in a data directory against metadata.json.
    
    Args:
        data_dir: Directory containing data and metadata.json
        
    Returns:
        dict: Verification results
    """
    metadata_file = data_dir / "metadata.json"
    
    if not metadata_file.exists():
        return {"status": "no_metadata", "dir": str(data_dir)}
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    stored_checksums = metadata.get("checksums", {})
    if not stored_checksums:
        stored_checksums = {metadata.get("file"): metadata.get("checksum")}
    
    results = {
        "dir": str(data_dir),
        "status": "ok",
        "files": {},
        "errors": []
    }
    
    # Verify each file
    for filename, expected_checksum in stored_checksums.items():
        if not filename:
            continue
            
        filepath = data_dir / filename
        
        if not filepath.exists():
            results["status"] = "missing_files"
            results["errors"].append(f"Missing: {filename}")
            results["files"][filename] = {"status": "missing"}
            continue
        
        actual_checksum = compute_checksum(filepath)
        
        if actual_checksum == expected_checksum:
            results["files"][filename] = {
                "status": "ok",
                "checksum": actual_checksum
            }
        else:
            results["status"] = "checksum_mismatch"
            results["errors"].append(f"Checksum mismatch: {filename}")
            results["files"][filename] = {
                "status": "mismatch",
                "expected": expected_checksum,
                "actual": actual_checksum
            }
    
    return results


def main():
    """Verify all downloaded data."""
    data_dirs = [
        Path("data/raw/sst"),
        Path("data/raw/sss"),
        Path("data/raw/chl"),
        Path("data/raw/fco2"),
    ]
    
    print("=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)
    
    all_ok = True
    
    for data_dir in data_dirs:
        print(f"\nVerifying: {data_dir}")
        
        if not data_dir.exists():
            print(f"  ✗ Directory does not exist")
            all_ok = False
            continue
        
        results = verify_directory(data_dir)
        
        if results["status"] == "no_metadata":
            print(f"  ⚠ No metadata.json found")
            all_ok = False
            continue
        
        if results["status"] == "ok":
            print(f"  ✓ All files verified ({len(results['files'])} files)")
        else:
            print(f"  ✗ Verification failed: {results['status']}")
            for error in results["errors"]:
                print(f"    - {error}")
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ ALL DATA VERIFIED")
    else:
        print("✗ VERIFICATION ERRORS FOUND")
    print("=" * 60)


if __name__ == "__main__":
    main()
