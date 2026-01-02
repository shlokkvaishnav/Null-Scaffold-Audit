from pathlib import Path
from typing import Union

import pandas as pd
import xarray as xr


class DataLoader:
    """Handles loading of raw NetCDF data and processed Parquet data."""

    def __init__(self, data_dir: Union[str, Path]):
        """
        Args:
            data_dir: Base directory for data.
        """
        self.data_dir = Path(data_dir)
        # Assumes data_dir is the PROJECT ROOT
        self.raw_path = (
            self.data_dir / "data" / "01_raw" / "SOCATv2025_tracks_gridded_monthly.nc"
        )
        self.processed_path = (
            self.data_dir / "data" / "03_processed" / "training_set.parquet"
        )

    def load_raw_dataset(self, chunks: dict = None) -> xr.Dataset:
        """
        Loads the raw NetCDF dataset using xarray.

        Args:
            chunks: Chunking dictionary for dask. Defaults to {"tmnth": 10}.

        Returns:
            The loaded xarray Dataset.
        """
        if chunks is None:
            chunks = {"tmnth": 10}

        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at {self.raw_path}")

        return xr.open_dataset(self.raw_path, chunks=chunks)

    def load_processed_dataframe(self) -> pd.DataFrame:
        """
        Loads the processed training data from Parquet.

        Returns:
            The loaded pandas DataFrame.
        """
        if not self.processed_path.exists():
            raise FileNotFoundError(
                f"Processed data not found at {self.processed_path}"
            )

        return pd.read_parquet(self.processed_path)
