"""Feature engineering utilities for SD-MoSE.

Functions for creating derived features from raw ocean data.
"""

import numpy as np
import pandas as pd
from typing import Optional


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived features for SD-MoSE model.
    
    Features added:
    - log_chl: log(chlorophyll) - handles skewness
    - lat_norm, lon_norm: normalized coordinates [-1, 1]
    - sin_month, cos_month: cyclic month encoding
    - year_norm: normalized year (centered at 2015)
    - sst_gradient: SST spatial gradient (if applicable)
    
    Args:
        df: DataFrame with raw features (sst, sss, chl, lat, lon, time)
        
    Returns:
        DataFrame with all derived features added
    """
    df = df.copy()
    
    # Log chlorophyll
    if 'chl' in df.columns:
        df['log_chl'] = np.log(np.clip(df['chl'], 1e-3, None))
    
    # Normalized coordinates
    if 'lat' in df.columns:
        df['lat_norm'] = df['lat'] / 90.0
    if 'lon' in df.columns:
        df['lon_norm'] = df['lon'] / 180.0
    
    # Cyclic time features
    if 'time' in df.columns:
        time_pd = pd.to_datetime(df['time'])
        df['year'] = time_pd.dt.year
        df['month'] = time_pd.dt.month
        
        month_angle = 2 * np.pi * (time_pd.dt.month - 1) / 12.0
        df['sin_month'] = np.sin(month_angle)
        df['cos_month'] = np.cos(month_angle)
        
        # Normalized year (centered at 2015)
        df['year_norm'] = (df['year'] - 2015) / 10.0
    
    return df


def add_cyclic_time(
    df: pd.DataFrame, 
    time_col: str = "time"
) -> pd.DataFrame:
    """Add sin/cos month encoding for seasonal cycles.
    
    Scientific rationale:
        sin/cos encoding ensures January and December are neighbors,
        and allows models to learn smooth seasonal patterns.
    """
    df = df.copy()
    
    if time_col in df.columns:
        time_pd = pd.to_datetime(df[time_col])
        month_angle = 2 * np.pi * (time_pd.dt.month - 1) / 12.0
        df['sin_month'] = np.sin(month_angle)
        df['cos_month'] = np.cos(month_angle)
    
    return df


def normalize_coordinates(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Normalize lat/lon to [-1, 1] range.
    
    Transformation:
        lat: [-90, 90] → [-1, 1]
        lon: [-180, 180] → [-1, 1]
    """
    df = df.copy()
    
    if lat_col in df.columns:
        df['lat_norm'] = df[lat_col] / 90.0
    if lon_col in df.columns:
        df['lon_norm'] = df[lon_col] / 180.0
    
    return df


def add_log_chlorophyll(
    df: pd.DataFrame,
    chl_col: str = "chl",
    epsilon: float = 1e-3,
) -> pd.DataFrame:
    """Add log-transformed chlorophyll.
    
    Scientific rationale:
        Chlorophyll-a is highly skewed (oligotrophic vs eutrophic).
        log(Chl) ~ Normal distribution, better for regression.
    """
    df = df.copy()
    
    if chl_col in df.columns:
        chl_safe = np.clip(df[chl_col].values, epsilon, None)
        df['log_chl'] = np.log(chl_safe)
    
    return df
