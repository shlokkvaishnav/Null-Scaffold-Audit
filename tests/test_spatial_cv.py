"""Unit tests for spatial cross-validation."""

import numpy as np
import pytest

from climate_discovery.validation.spatial_cv import SpatialCrossValidator, SpatialFold


@pytest.fixture
def sample_coordinates():
    """Create sample lat/lon data."""
    np.random.seed(42)
    n = 1000
    lats = np.random.uniform(-90, 90, n)
    lons = np.random.uniform(-180, 180, n)
    return lats, lons


def test_ocean_basin_split(sample_coordinates):
    """Test ocean basin splitting."""
    lats, lons = sample_coordinates
    
    cv = SpatialCrossValidator(strategy="ocean_basins")
    folds = cv.split(lats, lons)
    
    # Should have multiple folds (one per basin)
    assert len(folds) > 0
    
    # Each fold should be a SpatialFold
    for fold in folds:
        assert isinstance(fold, SpatialFold)
        assert len(fold.train_idx) > 0
        assert len(fold.test_idx) > 0


def test_no_overlap_in_folds(sample_coordinates):
    """Test that train and test sets don't overlap."""
    lats, lons = sample_coordinates
    
    cv = SpatialCrossValidator(strategy="ocean_basins")
    folds = cv.split(lats, lons)
    
    for fold in folds:
        train_set = set(fold.train_idx)
        test_set = set(fold.test_idx)
        
        # No overlap
        assert len(train_set & test_set) == 0


def test_all_samples_covered(sample_coordinates):
    """Test that all samples appear in at least one fold."""
    lats, lons = sample_coordinates
    n_samples = len(lats)
    
    cv = SpatialCrossValidator(strategy="ocean_basins")
    folds = cv.split(lats, lons)
    
    covered = set()
    for fold in folds:
        covered.update(fold.train_idx)
        covered.update(fold.test_idx)
    
    # All samples should be covered
    assert len(covered) == n_samples


def test_grid_blocks_split():
    """Test grid block splitting."""
    # Create uniform grid
    lats = np.repeat(np.linspace(-90, 90, 10), 10)
    lons = np.tile(np.linspace(-180, 180, 10), 10)
    
    cv = SpatialCrossValidator(strategy="grid_blocks", n_splits=4)
    folds = cv.split(lats, lons)
    
    assert len(folds) == 4


def test_latitude_bands_split():
    """Test latitude band splitting."""
    # Create data across all latitudes
    lats = np.linspace(-90, 90, 500)
    lons = np.random.uniform(-180, 180, 500)
    
    cv = SpatialCrossValidator(strategy="latitude_bands")
    folds = cv.split(lats, lons)
    
    # Should have folds for different latitude bands
    assert len(folds) > 0
    
    # Check that bands are distinct
    for fold in folds:
        test_lats = lats[fold.test_idx]
        # All test lats should be in same band
        lat_range = test_lats.max() - test_lats.min()
        assert lat_range < 100  # Within a band


def test_basin_assignment():
    """Test ocean basin assignment logic."""
    cv = SpatialCrossValidator(strategy="ocean_basins")
    
    # Test specific points
    assert cv._assign_ocean_basin(70, 0) == "Arctic"
    assert cv._assign_ocean_basin(-70, 0) == "Southern"
    assert cv._assign_ocean_basin(0, -30) == "Atlantic"
    assert cv._assign_ocean_basin(0, 100) == "Indian"
    assert cv._assign_ocean_basin(0, 170) == "Pacific"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
