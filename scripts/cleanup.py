"""Clean up temporary and cache files from the project.

This script removes:
- Python cache files (__pycache__, *.pyc)
- Build artifacts (*.egg-info, dist/, build/)
- PySR temporary files (pysr_tmp/)
- Log files

Run this after completing experiments to keep the repository clean.
"""

import os
import shutil
from pathlib import Path

def remove_pycache():
    """Remove all __pycache__ directories."""
    count = 0
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_dir)
                print(f"✓ Removed: {cache_dir}")
                count += 1
            except Exception as e:
                print(f"✗ Failed to remove {cache_dir}: {e}")
    return count

def remove_pyc_files():
    """Remove all .pyc files."""
    count = 0
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"✓ Removed: {file_path}")
                    count += 1
                except Exception as e:
                    print(f"✗ Failed to remove {file_path}: {e}")
    return count

def remove_build_artifacts():
    """Remove build artifacts."""
    artifacts = [
        'src/climate_discovery.egg-info',
        'build',
        'dist',
        '.eggs',
    ]
    count = 0
    for artifact in artifacts:
        if os.path.exists(artifact):
            try:
                if os.path.isdir(artifact):
                    shutil.rmtree(artifact)
                else:
                    os.remove(artifact)
                print(f"✓ Removed: {artifact}")
                count += 1
            except Exception as e:
                print(f"✗ Failed to remove {artifact}: {e}")
    return count

def remove_pysr_temp():
    """Remove PySR temporary files."""
    if os.path.exists('pysr_tmp'):
        try:
            shutil.rmtree('pysr_tmp')
            print(f"✓ Removed: pysr_tmp/")
            return 1
        except Exception as e:
            print(f"✗ Failed to remove pysr_tmp: {e}")
    return 0

def remove_log_files():
    """Remove .log files."""
    count = 0
    for root, dirs, files in os.walk('.'):
        # Skip data and results directories
        if 'data' in root or 'results' in root:
            continue
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"✓ Removed: {file_path}")
                    count += 1
                except Exception as e:
                    print(f"✗ Failed to remove {file_path}: {e}")
    return count

def main():
    print("=" * 70)
    print("SD-MoSE Project Cleanup")
    print("=" * 70)
    print()
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    print("🧹 Removing Python cache files...")
    pycache_count = remove_pycache()
    pyc_count = remove_pyc_files()
    print(f"   Removed {pycache_count} __pycache__ directories, {pyc_count} .pyc files\n")
    
    print("🧹 Removing build artifacts...")
    build_count = remove_build_artifacts()
    print(f"   Removed {build_count} build artifacts\n")
    
    print("🧹 Removing PySR temporary files...")
    pysr_count = remove_pysr_temp()
    print(f"   Removed {pysr_count} PySR temp directories\n")
    
    print("🧹 Removing log files...")
    log_count = remove_log_files()
    print(f"   Removed {log_count} log files\n")
    
    total = pycache_count + pyc_count + build_count + pysr_count + log_count
    
    print("=" * 70)
    print(f"✅ Cleanup complete! Removed {total} items total.")
    print("=" * 70)
    print()
    print("Note: Data files, results, and checkpoints are preserved.")
    print("Run 'git status' to verify changes.")

if __name__ == "__main__":
    main()
