"""Core utilities: feature engineering functions.

Extracted from utils.py for better organization.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def add_cyclic_time_features(
    df: pd.DataFrame, 
    time_col: str = "time"
) -> pd.DataFrame:
    """Add sin/cos encodings of month for seasonal cycles.
    
    Args:
        df: DataFrame with datetime column
        time_col: Name of time column
        
    Returns:
        DataFrame with added columns: sin_month, cos_month
        
    Scientific rationale:
        sin/cos encoding ensures January and December are neighbors,
        and allows neural networks to learn seasonal periodicity.
    """
    df = df.copy()
    
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found")
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Month as angle: Jan=0, Dec=11 → [0, 2π)
    month_numeric = df[time_col].dt.month - 1  # 0-11
    angle = 2 * np.pi * month_numeric / 12.0
    
    df["sin_month"] = np.sin(angle)
    df["cos_month"] = np.cos(angle)
    
    return df


def add_normalized_year(
    df: pd.DataFrame,
    time_col: str = "time",
    reference_year: int = 2015,
) -> pd.DataFrame:
    """Add normalized year feature for long-term trends.
    
    Args:
        df: DataFrame with datetime column
        time_col: Name of time column
        reference_year: Center year for normalization
        
    Returns:
        DataFrame with added 'year_norm' column
        
    Note:
        year_norm = (year - reference_year) / 10.0
        This keeps values in [-1, 1] range for ~2005-2025
    """
    df = df.copy()
    
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found")
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    year = df[time_col].dt.year
    df["year_norm"] = (year - reference_year) / 10.0
    
    return df


def add_log_chlorophyll(
    df: pd.DataFrame,
    chl_col: str = "chl",
    epsilon: float = 1e-3,
) -> pd.DataFrame:
    """Add log-transformed chlorophyll (handles skewness).
    
    Args:
        df: DataFrame with chlorophyll column (mg/m³)
        chl_col: Name of chlorophyll column
        epsilon: Floor value to prevent log(0)
        
    Returns:
        DataFrame with added 'log_chl' column
        
    Scientific rationale:
        Chlorophyll-a is highly skewed (oligotrophic vs eutrophic).
        log(Chl) ~ Normal distribution, better for regression.
    """
    df = df.copy()
    
    if chl_col not in df.columns:
        raise KeyError(f"Chlorophyll column '{chl_col}' not found")
    
    # Clip to positive values
    chl_safe = np.clip(df[chl_col].values, epsilon, None)
    df["log_chl"] = np.log(chl_safe)
    
    return df


def normalize_coordinates(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Normalize lat/lon to [-1, 1] for neural network stability.
    
    Args:
        df: DataFrame with lat/lon columns
        lat_col: Latitude column name
        lon_col: Longitude column name
        
    Returns:
        DataFrame with 'lat_norm', 'lon_norm' columns
        
    Transformation:
        lat: [-90, 90] → [-1, 1]
        lon: [-180, 180] → [-1, 1]
    """
    df = df.copy()
    
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(f"Missing coordinate columns: {lat_col}, {lon_col}")
    
    df["lat_norm"] = df[lat_col] / 90.0
    df["lon_norm"] = df[lon_col] / 180.0
    
    return df
