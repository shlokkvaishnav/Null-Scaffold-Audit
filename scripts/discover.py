import argparse
import logging
from pathlib import Path

from climate_discovery.data import DataLoader
from climate_discovery.models import SymbolicDiscovery

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run equation discovery.")
    parser.add_argument(
        "--data_dir", type=str, default=".", help="Base project directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pysr_config.yaml",
        help="Path to PySR config",
    )
    parser.add_argument(
        "--sample_size", type=int, default=25000, help="Number of samples to use"
    )
    args = parser.parse_args()

    # Load Data
    loader = DataLoader(args.data_dir)
    try:
        df = loader.load_processed_dataframe()
    except FileNotFoundError:
        logger.error(
            "Processed data not found. Run scripts/run_preprocessing.py first."
        )
        return

    # Subsample
    if len(df) > args.sample_size:
        logger.info(f"Subsampling to {args.sample_size} points...")
        df = df.sample(n=args.sample_size, random_state=42)

    # Run Discovery
    logger.info(f"🧠 Initializing PySR with config from {args.config}...")
    try:
        discovery = SymbolicDiscovery.from_yaml(args.config)

        logger.info("🚀 Starting Discovery Loop...")
        r2 = discovery.fit(df)

        logger.info("=" * 40)
        logger.info("🏆 THE DISCOVERED EQUATION")
        logger.info("=" * 40)
        logger.info(discovery.get_best_equation())
        logger.info(f"✅ Validation R^2 Score: {r2:.4f}")

    except FileNotFoundError:
        logger.error(f"Config file not found at {args.config}")
    except Exception as e:
        logger.error(f"An error occurred during discovery: {e}")


if __name__ == "__main__":
    main()
