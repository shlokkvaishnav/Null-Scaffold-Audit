import requests
from tqdm import tqdm
from pathlib import Path

#Configuration
DATA_URL = "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0304549/SOCATv2025_Gridded_Data/SOCATv2025_tracks_gridded_monthly.nc"
RAW_DIR = Path("data/01_raw")
FILENAME = "SOCATv2025_tracks_gridded_monthly.nc"

def download_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DIR / FILENAME

    #Check existing file size for resume
    if file_path.exists():
        resume_byte_pos = file_path.stat().st_size
    else:
        resume_byte_pos = 0

    #Set headers to request only the missing bytes
    headers = {}
    if resume_byte_pos > 0:
        headers = {"Range": f"bytes={resume_byte_pos}-"}
        mode = "ab"
        print(f"🔄 Resuming download from {resume_byte_pos / (1024**3):.2f} GB...")
    else:
        mode = "wb"
        print(f"⬇️  Starting new download from: {DATA_URL}")

    #Stream with Progress Bar
    try:
        response = requests.get(DATA_URL, headers=headers, stream=True)
        
        # Handle case where server refuses partial content (HTTP 206)
        if response.status_code == 416:
            print("✅ File is already complete!")
            return
        
        total_size = int(response.headers.get('content-length', 0)) + resume_byte_pos

        with open(file_path, mode) as file, tqdm(
            desc=FILENAME,
            total=total_size,
            initial=resume_byte_pos, # Start bar at resumed position
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024*1024*8):
                if chunk:
                    bar.update(len(chunk))
                    file.write(chunk)

        print(f"\n✅ Download Complete: {file_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    download_data()