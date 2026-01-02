from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_discovery.data import DataLoader, DataPreprocessor


def test_data_processor_methods():
    # Create dummy DataFrame
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=5, freq="ME"),
            "lat": [10, -20, 30, -40, 50],
            "lon": [0, 0, 0, 0, 0],
            "fCO2": [350, 360, 370, 380, 390],
            "SST": [25, 20, 15, 10, 5],
            "Salinity": [35, 35, 35, 35, 35],
        }
    )

    # Test adding features
    df_feat = DataPreprocessor.add_features(df)
    assert "Year" in df_feat.columns
    assert "AbsLat" in df_feat.columns
    assert df_feat["AbsLat"].iloc[1] == 20

    # Test guardrails (inject bad value)
    df_feat.loc[0, "SST"] = -100  # Invalid
    df_clean = DataPreprocessor.clean_and_validate(df_feat)
    assert len(df_clean) == 4  # Should drop one row


def test_loader_path_resolution(tmp_path):
    # Mock project root
    loader = DataLoader(tmp_path)
    expected_raw = tmp_path / "data" / "01_raw" / "SOCATv2025_tracks_gridded_monthly.nc"
    assert loader.raw_path == expected_raw
