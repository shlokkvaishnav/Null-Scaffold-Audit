"""Spatial Feature Engineering Module

Adds spatial features to address autocorrelation identified in residual analysis.
This helps reduce the geographic clustering of residuals.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class SpatialFeatureEngineer:
    """Engineer spatial features for ocean carbon modeling.
    
    Features added:
    - Ocean basin indicators (Atlantic, Pacific, Indian)
    - Latitude bands (Tropical, Temperate, Polar)
    - Haversine distance to key oceanographic features
    - Coastal proximity indicators
    """
    
    def __init__(self):
        self.basin_boundaries = {
            'atlantic': {'lon': (-80, 20), 'lat': (-60, 70)},
            'pacific': {'lon': (120, -80), 'lat': (-60, 65)},
            'indian': {'lon': (20, 120), 'lat': (-60, 30)},
        }
        
        self.lat_bands = {
            'tropical': (-20, 20),
            'temperate_n': (20, 50),
            'temperate_s': (-50, -20),
            'polar_n': (50, 90),
            'polar_s': (-90, -50),
        }
        
    def add_ocean_basin_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add one-hot encoded ocean basin indicators.
        
        Args:
            df: DataFrame with 'lat' and 'lon' columns
            
        Returns:
            DataFrame with basin_atlantic, basin_pacific, basin_indian columns
        """
        df = df.copy()
        
        # Initialize all to False
        df['basin_atlantic'] = 0
        df['basin_pacific'] = 0
        df['basin_indian'] = 0
        
        for _, row in df.iterrows():
            lat, lon = row['lat'], row['lon']
            
            # Check Atlantic
            if self._in_basin(lat, lon, 'atlantic'):
                df.loc[row.name, 'basin_atlantic'] = 1
            # Check Pacific
            elif self._in_basin(lat, lon, 'pacific'):
                df.loc[row.name, 'basin_pacific'] = 1
            # Check Indian
            elif self._in_basin(lat, lon, 'indian'):
                df.loc[row.name, 'basin_indian'] = 1
                
        return df
    
    def add_latitude_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add latitude band categorical features.
        
        Args:
            df: DataFrame with 'lat' column
            
        Returns:
            DataFrame with band_tropical, band_temperate, band_polar columns
        """
        df = df.copy()
        
        df['band_tropical'] = 0
        df['band_temperate'] = 0
        df['band_polar'] = 0
        
        lat = df['lat'].values
        
        # Tropical
        mask = (lat >= -20) & (lat <= 20)
        df.loc[mask, 'band_tropical'] = 1
        
        # Temperate
        mask = ((lat >= 20) & (lat <= 50)) | ((lat >= -50) & (lat <= -20))
        df.loc[mask, 'band_temperate'] = 1
        
        # Polar
        mask = (lat > 50) | (lat < -50)
        df.loc[mask, 'band_polar'] = 1
        
        return df
    
    def add_haversine_distance(
        self, 
        df: pd.DataFrame, 
        ref_lat: float, 
        ref_lon: float,
        feature_name: str = 'dist_km'
    ) -> pd.DataFrame:
        """Add distance (km) to a reference point using Haversine formula.
        
        Args:
            df: DataFrame with 'lat' and 'lon' columns
            ref_lat: Reference latitude
            ref_lon: Reference longitude
            feature_name: Name for the distance feature
            
        Returns:
            DataFrame with added distance column
        """
        df = df.copy()
        
        lat1 = np.radians(df['lat'].values)
        lon1 = np.radians(df['lon'].values)
        lat2 = np.radians(ref_lat)
        lon2 = np.radians(ref_lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        R = 6371.0
        distance = R * c
        
        df[feature_name] = distance
        return df
    
    def add_equator_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add distance from equator (simple latitude-based).
        
        Args:
            df: DataFrame with 'lat' column
            
        Returns:
            DataFrame with 'dist_equator' column (in degrees)
        """
        df = df.copy()
        df['dist_equator'] = np.abs(df['lat'])
        return df
    
    def add_all_spatial_features(
        self, 
        df: pd.DataFrame,
        include_basins: bool = True,
        include_bands: bool = True,
        include_distances: bool = True
    ) -> pd.DataFrame:
        """Add all spatial features at once.
        
        Args:
            df: DataFrame with 'lat' and 'lon' columns
            include_basins: Add ocean basin indicators
            include_bands: Add latitude band indicators
            include_distances: Add distance features
            
        Returns:
            DataFrame with all selected spatial features
        """
        if include_basins:
            df = self.add_ocean_basin_indicators(df)
            
        if include_bands:
            df = self.add_latitude_bands(df)
            
        if include_distances:
            df = self.add_equator_distance(df)
            
        return df
    
    def _in_basin(self, lat: float, lon: float, basin: str) -> bool:
        """Check if coordinates are in specified ocean basin."""
        bounds = self.basin_boundaries[basin]
        
        # Handle longitude wrapping for Pacific
        if basin == 'pacific':
            # Pacific crosses dateline: 120E to -80W (or 280E)
            in_lon = lon >= 120 or lon <= -80
        else:
            in_lon = bounds['lon'][0] <= lon <= bounds['lon'][1]
            
        in_lat = bounds['lat'][0] <= lat <= bounds['lat'][1]
        
        return in_lon and in_lat


def demonstrate_spatial_features():
    """Demonstrate spatial feature engineering."""
    # Create sample data
    sample_data = pd.DataFrame({
        'lat': [0, 35, -45, 70, -10, 25],
        'lon': [-30, 150, 80, -170, -60, 10],
        'sst': [28, 15, 8, 2, 27, 20],
        'fco2': [380, 350, 320, 300, 400, 360]
    })
    
    print("Original data:")
    print(sample_data)
    
    # Add spatial features
    engineer = SpatialFeatureEngineer()
    enriched = engineer.add_all_spatial_features(sample_data)
    
    print("\n\nEnriched with spatial features:")
    print(enriched)
    
    print("\n\nNew spatial features:")
    spatial_cols = [c for c in enriched.columns if c not in sample_data.columns]
    print(spatial_cols)


if __name__ == "__main__":
    demonstrate_spatial_features()
