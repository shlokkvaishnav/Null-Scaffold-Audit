import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List

class DataPreprocessor:
    """Handles preprocessing and feature engineering for climate data."""

    DEFAULT_VARS = {
        "fco2_ave_unwtd": "fCO2",
        "sst_ave_unwtd": "SST",
        "salinity_ave_unwtd": "Salinity"
    }

    @staticmethod
    def flatten_dataset(ds: xr.Dataset, vars_map: Dict[str, str] = None) -> pd.DataFrame:
        """
        Renames variables and flattens an xarray Dataset to a pandas DataFrame.

        Args:
            ds: Input xarray Dataset.
            vars_map: Dictionary mapping raw variable names to target names.

        Returns:
            Flattened pandas DataFrame.
        """
        if vars_map is None:
            vars_map = DataPreprocessor.DEFAULT_VARS
            
        # Rename dims
        ds = ds.rename({"tmnth": "time", "ylat": "lat", "xlon": "lon"})
        
        # Select and rename variables
        subset = ds[list(vars_map.keys())]
        subset = subset.rename(vars_map)
        
        return subset.to_dataframe().reset_index()

    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds derived features like decimal year and absolute latitude.

        Args:
            df: Input DataFrame with 'time' and 'lat' columns.

        Returns:
            DataFrame with added features.
        """
        df = df.copy()
        
        # Time: Decimal Year
        df['Year'] = df['time'].dt.year + df['time'].dt.dayofyear / 365.25
        
        # Space: Absolute Latitude
        df['AbsLat'] = np.abs(df['lat'])
        
        return df

    @staticmethod
    def clean_and_validate(df: pd.DataFrame, required_cols: List[str] = None) -> pd.DataFrame:
        """
        Removes missing values and enforces physical constraints.

        Args:
            df: Input DataFrame.
            required_cols: List of columns that must not be null.

        Returns:
            Cleaned DataFrame.
        """
        if required_cols is None:
            required_cols = ["fCO2", "SST", "Salinity", "Year", "AbsLat"]
            
        clean_df = df.dropna(subset=required_cols)
        
        # Physics Guardrails
        clean_df = clean_df[
            (clean_df["SST"] > -2) & (clean_df["SST"] < 35) &
            (clean_df["Salinity"] > 20) & (clean_df["Salinity"] < 40)
        ]
        
        return clean_df
