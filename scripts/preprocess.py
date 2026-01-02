import argparse
import logging
from pathlib import Path

from climate_discovery.data import DataLoader, DataPreprocessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process climate data.")
    parser.add_argument(
        "--data_dir", type=str, default=".", help="Base project directory"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    loader = DataLoader(data_dir=data_dir)

    logger.info("⏳ Loading NetCDF...")
    try:
        ds = loader.load_raw_dataset()
    except FileNotFoundError as e:
        logger.error(e)
        logger.info(
            "Ensure you have run the downloader script or placed the .nc file correctly."
        )
        return

    logger.info("🚜 Flattening to table...")
    df = DataPreprocessor.flatten_dataset(ds)

    logger.info("Feature Engineering: Adding Time & Space...")
    df = DataPreprocessor.add_features(df)

    logger.info("🧹 Cleaning missing values and applying physics guardrails...")
    clean_df = DataPreprocessor.clean_and_validate(df)

    logger.info(f"✅ CLEAN DATA POINTS: {len(clean_df):,}")

    output_path = loader.processed_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(output_path)
    logger.info(f"💾 Saved processed data to {output_path}")


if __name__ == "__main__":
    main()
