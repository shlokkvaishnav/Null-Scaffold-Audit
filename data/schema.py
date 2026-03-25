"""
Data Contract - Frozen Observation Schema

This schema is SACRED. Once frozen, features cannot be added or removed.
All preprocessing must respect this contract.
"""

from typing import Dict, List, Any, TypedDict
from datetime import datetime


class SpatialResolution(TypedDict):
    """Spatial grid specification."""
    grid_type: str  # "regular_lat_lon"
    lat_min: float
    lat_max: float
    lat_step: float
    lon_min: float
    lon_max: float
    lon_step: float


class TemporalResolution(TypedDict):
    """Temporal sampling specification."""
    frequency: str  # "monthly"
    start_date: str
    end_date: str


class DataContract:
    """
    Canonical data contract for SD-MoSE agent.
    
    This contract is IMMUTABLE once Phase D begins.
    """
    
    # Feature set (inputs)
    FEATURES: List[str] = [
        "sst",   # Sea Surface Temperature (°C)
        "sss",   # Sea Surface Salinity (PSU)
        "chl",   # Chlorophyll-a concentration (mg/m³, log-transformed)
    ]
    
    # Target variable (output)
    TARGET: str = "fco2"  # Surface ocean fCO2 (μatm)
    
    # Coordinate dimensions
    COORDINATES: List[str] = ["time", "lat", "lon"]
    
    # Spatial resolution
    SPATIAL: SpatialResolution = {
        "grid_type": "regular_lat_lon",
        "lat_min": -88.0,
        "lat_max": 88.0,
        "lat_step": 2.0,  # 2° resolution
        "lon_min": -178.0,
        "lon_max": 178.0,
        "lon_step": 2.0,
    }
    
    # Temporal resolution
    TEMPORAL: TemporalResolution = {
        "frequency": "monthly",
        "start_date": "2010-01-01",
        "end_date": "2020-12-31",
    }
    
    # Train/test split (time-based)
    TRAIN_PERIOD: tuple = ("2010-01-01", "2017-12-31")  # 96 months
    TEST_PERIOD: tuple = ("2018-01-01", "2020-12-31")   # 36 months
    
    # Expected dimensions
    N_TIME: int = 132  # 2010-2020 monthly
    N_LAT: int = 89    # (-88 to 88) / 2 + 1
    N_LON: int = 180   # (-178 to 178) / 2 + 1
    N_FEATURES: int = len(FEATURES)
    
    # Data sources (for reference)
    SOURCES: Dict[str, Dict[str, str]] = {
        "sst": {
            "name": "NOAA OISST v2.1",
            "url": "https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html",
            "variable": "sst",
            "units": "°C",
            "native_resolution": "0.25° daily",
        },
        "sss": {
            "name": "EN4.2.2",
            "url": "https://www.metoffice.gov.uk/hadobs/en4/",
            "variable": "salinity",
            "units": "PSU",
            "native_resolution": "1° monthly",
        },
        "chl": {
            "name": "NASA MODIS-Aqua Level 3",
            "url": "https://oceancolor.gsfc.nasa.gov/",
            "variable": "chlor_a",
            "units": "mg/m³",
            "native_resolution": "9km monthly",
        },
        "fco2": {
            "name": "SOCAT Gridded v2023",
            "url": "https://www.socat.info/",
            "variable": "fco2_sw_mean",
            "units": "μatm",
            "native_resolution": "1° monthly",
        },
    }
    
    @classmethod
    def validate_dataset(cls, dataset) -> bool:
        """
        Validate that a dataset conforms to this contract.
        
        Args:
            dataset: xarray.Dataset
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If validation fails
        """
        import xarray as xr
        
        if not isinstance(dataset, xr.Dataset):
            raise ValueError("Dataset must be xarray.Dataset")
        
        # Check required variables
        required_vars = ["features", "target", "mask"]
        missing = set(required_vars) - set(dataset.data_vars)
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Check dimensions
        required_dims = ["time", "lat", "lon", "feature"]
        missing_dims = set(required_dims) - set(dataset.dims)
        if missing_dims:
            raise ValueError(f"Missing required dimensions: {missing_dims}")
        
        # Check dimension sizes
        if dataset.dims["time"] != cls.N_TIME:
            raise ValueError(f"Expected {cls.N_TIME} time steps, got {dataset.dims['time']}")
        
        if dataset.dims["feature"] != cls.N_FEATURES:
            raise ValueError(f"Expected {cls.N_FEATURES} features, got {dataset.dims['feature']}")
        
        # Check no NaNs in masked regions
        mask = dataset["mask"].values
        features_masked = dataset["features"].values
        target_masked = dataset["target"].values
        
        import numpy as np
        if np.any(np.isnan(features_masked[mask])):
            raise ValueError("NaNs found in masked feature regions")
        
        if np.any(np.isnan(target_masked[mask])):
            raise ValueError("NaNs found in masked target regions")
        
        return True
    
    @classmethod
    def get_grid_spec(cls) -> Dict[str, Any]:
        """Get grid specification for regridding."""
        import numpy as np
        
        lat = np.arange(
            cls.SPATIAL["lat_min"],
            cls.SPATIAL["lat_max"] + cls.SPATIAL["lat_step"],
            cls.SPATIAL["lat_step"]
        )
        lon = np.arange(
            cls.SPATIAL["lon_min"],
            cls.SPATIAL["lon_max"] + cls.SPATIAL["lon_step"],
            cls.SPATIAL["lon_step"]
        )
        
        return {"lat": lat, "lon": lon}
    
    @classmethod
    def export_metadata(cls) -> Dict[str, Any]:
        """Export contract as metadata dict."""
        return {
            "schema_version": "1.0",
            "created": datetime.now().isoformat(),
            "features": cls.FEATURES,
            "target": cls.TARGET,
            "coordinates": cls.COORDINATES,
            "spatial_resolution": cls.SPATIAL,
            "temporal_resolution": cls.TEMPORAL,
            "train_period": cls.TRAIN_PERIOD,
            "test_period": cls.TEST_PERIOD,
            "sources": cls.SOURCES,
        }


# Convenience aliases
FEATURES = DataContract.FEATURES
TARGET = DataContract.TARGET
N_FEATURES = DataContract.N_FEATURES
