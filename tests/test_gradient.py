"""Unit tests for SST gradient computation."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_discovery.utils import compute_sst_gradient


@pytest.fixture
def simple_grid_data():
    """Create simple test grid data."""
    lats = np.array([0, 1, 2, 3, 4])
    lons = np.array([0, 1, 2, 3, 4])
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    
    # Create SST field with known gradient
    sst = np.zeros((len(times), len(lats), len(lons)))
    for t in range(len(times)):
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                sst[t, i, j] = lat + lon  # Linear gradient
    
    ds = xr.Dataset(
        {
            "sst": (["time", "lat", "lon"], sst),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    
    return ds


@pytest.fixture
def frontal_data():
    """Create data with sharp SST front."""
    lats = np.linspace(-10, 10, 20)
    lons = np.linspace(-10, 10, 20)
    times = pd.date_range("2020-01-01", periods=1)
    
    # Create front: cold on left, warm on right
    sst = np.zeros((1, len(lats), len(lons)))
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if lon < 0:
                sst[0, i, j] = 5.0  # Cold
            else:
                sst[0, i, j] = 25.0  # Warm
    
    ds = xr.Dataset(
        {
            "sst": (["time", "lat", "lon"], sst),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    
    return ds


def test_sst_gradient_non_negative(simple_grid_data):
    """Test that SST gradient is always non-negative."""
    result = compute_sst_gradient(simple_grid_data)
    
    assert "sst_gradient" in result
    assert (result["sst_gradient"].values >= 0).all(), "SST gradient should be non-negative"


def test_sst_gradient_shape(simple_grid_data):
    """Test that output shape matches input."""
    result = compute_sst_gradient(simple_grid_data)
    
    assert result["sst_gradient"].shape == simple_grid_data["sst"].shape


def test_sst_gradient_at_fronts(frontal_data):
    """Test that gradients are high at fronts."""
    result = compute_sst_gradient(frontal_data)
    
    # Get gradients near the front (lon ~ 0)
    lon_idx = len(frontal_data.lon) // 2
    front_gradients = result["sst_gradient"].isel(time=0, lon=slice(lon_idx-2, lon_idx+2))
    
    # Get gradients away from front
    far_gradients = result["sst_gradient"].isel(time=0, lon=slice(0, 5))
    
    # Gradient at front should be much higher
    assert front_gradients.mean() > far_gradients.mean(), "Gradient should be higher at fronts"


def test_sst_gradient_zero_for_constant(simple_grid_data):
    """Test that gradient is zero for constant SST."""
    # Set constant SST
    simple_grid_data["sst"].values[:] = 20.0
    
    result = compute_sst_gradient(simple_grid_data)
    
    # Gradient should be near zero (allowing for numerical errors)
    assert np.allclose(result["sst_gradient"].values, 0, atol=1e-6), \
        "Gradient should be zero for constant SST"


def test_sst_gradient_with_nan():
    """Test that NaN values are handled correctly."""
    lats = np.array([0, 1, 2])
    lons = np.array([0, 1, 2])
    times = pd.date_range("2020-01-01", periods=1)
    
    sst = np.ones((1, 3, 3)) * 15.0
    sst[0, 1, 1] = np.nan  # Add NaN in center
    
    ds = xr.Dataset(
        {
            "sst": (["time", "lat", "lon"], sst),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    
    result = compute_sst_gradient(ds)
    
    # Should not raise error
    assert "sst_gradient" in result
    # NaN values should propagate or be handled
    assert result["sst_gradient"].shape == (1, 3, 3)


def test_sst_gradient_dimensional_consistency():
    """Test that gradient has correct physical units."""
    # Create data with known spacing
    lats = np.array([0, 1, 2])  # 1 degree spacing
    lons = np.array([0, 1, 2])
    times = pd.date_range("2020-01-01", periods=1)
    
    # SST increases by 1°C per degree latitude
    sst = np.zeros((1, 3, 3))
    for i in range(3):
        sst[0, i, :] = i * 1.0
    
    ds = xr.Dataset(
        {
            "sst": (["time", "lat", "lon"], sst),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    
    result = compute_sst_gradient(ds)
    
    # Gradient should be approximately 1.0 (°C per degree)
    # (allowing for Earth curvature effects)
    mean_grad = result["sst_gradient"].mean().values
    assert 0.5 < mean_grad < 2.0, f"Expected gradient ~1, got {mean_grad}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
