"""Download SOCAT (physics) and Copernicus chlorophyll (biology) data."""
import argparse
import logging
from pathlib import Path

import requests
import copernicusmarine
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# NCEI OCADS 0304549. Try v2025; v2024 often 404.
SOCAT_URLS = [
    "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0304549/SOCATv2025_Gridded_Data/SOCATv2025_tracks_gridded_monthly.nc",
    "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0304549/SOCATv2024_Gridded_Data/SOCATv2024_tracks_gridded_monthly.nc",
]
SOCAT_FILENAME = "SOCATv2025_tracks_gridded_monthly.nc"
CMEMS_DATASET_ID = "cmems_mod_glo_bgc_my_0.25deg_P1M-m"
CMEMS_FILENAME = "cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc"
CMEMS_VARIABLE = "chl"
START_DATE = "1993-01-01"
END_DATE = "2024-12-31"


def check_copernicus_auth():
    try:
        logger.info("Verifying Copernicus credentials...")
    except Exception:
        pass


def download_socat(output_dir: Path) -> bool:
    file_path = output_dir / SOCAT_FILENAME
    if file_path.exists() and file_path.stat().st_size > 100_000_000:
        logger.info("SOCAT already present: %s", file_path)
        return True
    for url in SOCAT_URLS:
        label = url.split("/")[-2]
        try:
            logger.info("Trying %s ...", label)
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(file_path, "wb") as f, tqdm(
                desc="SOCAT", total=total, unit="iB", unit_scale=True, unit_divisor=1024
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        bar.update(len(chunk))
                        f.write(chunk)
            logger.info("SOCAT saved: %s", file_path)
            return True
        except Exception as e:
            logger.warning("SOCAT download failed (%s): %s", label, e)
            if file_path.exists():
                file_path.unlink(missing_ok=True)
    logger.error("SOCAT download failed for all URLs.")
    return False


def download_chlorophyll(output_dir: Path) -> bool:
    file_path = output_dir / CMEMS_FILENAME
    if file_path.exists():
        logger.info("Chlorophyll already exists: %s", file_path)
        return True
    logger.info("Downloading Copernicus chlorophyll (%s to %s)...", START_DATE, END_DATE)
    try:
        copernicusmarine.subset(
            dataset_id=CMEMS_DATASET_ID,
            variables=[CMEMS_VARIABLE],
            start_datetime=START_DATE,
            end_datetime=END_DATE,
            minimum_longitude=-180,
            maximum_longitude=180,
            minimum_latitude=-90,
            maximum_latitude=90,
            minimum_depth=0,
            maximum_depth=10,
            output_directory=str(output_dir),
            output_filename=CMEMS_FILENAME,
            overwrite=True,
        )
        logger.info("Chlorophyll saved: %s", file_path)
        return True
    except Exception as e:
        logger.error("Copernicus error: %s", e)
        logger.warning("Run: copernicusmarine login")
        return False


def main():
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from climate_discovery.config import RAW_DIR

    ap = argparse.ArgumentParser(description="Download climate data.")
    ap.add_argument("--data_dir", type=str, default=None, help="Raw data dir (default: config RAW_DIR)")
    args = ap.parse_args()
    out = Path(args.data_dir) if args.data_dir else RAW_DIR
    out.mkdir(parents=True, exist_ok=True)
    check_copernicus_auth()
    ok_socat = download_socat(out)
    ok_chl = download_chlorophyll(out)
    if not ok_socat:
        logger.error("SOCAT required. Fix URLs or download manually to %s", out / SOCAT_FILENAME)
        sys.exit(1)
    if not ok_chl:
        logger.error("Chlorophyll required. Run: copernicusmarine login")
        sys.exit(1)


if __name__ == "__main__":
    main()
