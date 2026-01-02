import argparse
import logging
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DATA_URL = "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0304549/SOCATv2025_Gridded_Data/SOCATv2025_tracks_gridded_monthly.nc"
FILENAME = "SOCATv2025_tracks_gridded_monthly.nc"


def download_data(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / FILENAME

    logger.info(f"Target path: {file_path}")

    # Check existing file size for resume
    if file_path.exists():
        resume_byte_pos = file_path.stat().st_size
    else:
        resume_byte_pos = 0

    headers = {}
    mode = "wb"
    if resume_byte_pos > 0:
        headers = {"Range": f"bytes={resume_byte_pos}-"}
        mode = "ab"
        logger.info(
            f"🔄 Resuming download from {resume_byte_pos / (1024**3):.2f} GB..."
        )
    else:
        logger.info(f"⬇️  Starting new download from: {DATA_URL}")

    try:
        response = requests.get(DATA_URL, headers=headers, stream=True)

        # Handle case where server refuses partial content (HTTP 206)
        if response.status_code == 416:
            logger.info("✅ File is already complete!")
            return

        total_size = int(response.headers.get("content-length", 0)) + resume_byte_pos

        with (
            open(file_path, mode) as file,
            tqdm(
                desc=FILENAME,
                total=total_size,
                initial=resume_byte_pos,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024 * 8):
                if chunk:
                    bar.update(len(chunk))
                    file.write(chunk)

        logger.info(f"✅ Download Complete: {file_path}")

    except Exception as e:
        logger.error(f"❌ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download climate data.")
    parser.add_argument(
        "--data_dir", type=str, default="data/01_raw", help="Directory to save data"
    )
    args = parser.parse_args()

    download_data(Path(args.data_dir))
