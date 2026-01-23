"""PyTorch Dataset classes for SD-MoSE training.

Provides efficient data loading with:
- On-the-fly feature extraction
- NaN handling
- Spatial/temporal indexing
- Memory-mapped NetCDF for large datasets
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


class ClimateDataset(Dataset):
    """PyTorch Dataset for ocean climate data.
    
    Loads preprocessed NetCDF files and provides (X_expert, X_gate, y) tuples.
    
    Args:
        netcdf_path: Path to preprocessed dataset
        expert_features: Features for symbolic experts
        gating_features: Features for gating network
        target: Target variable name
        drop_nan: Remove samples with NaN values
        
    Example:
        >>> dataset = ClimateDataset(
        ...     "data/processed/train_dataset.nc",
        ...     expert_features=["sst", "sss", "log_chl"],
        ...     gating_features=["lat_norm", "lon_norm", "sst"],
        ...     target="fco2"
        ... )
        >>> X_expert, X_gate, y = dataset[0]
    """
    
    def __init__(
        self,
        netcdf_path: str | Path,
        expert_features: List[str],
        gating_features: List[str],
        target: Optional[str] = None,
        drop_nan: bool = True,
    ):
        self.netcdf_path = Path(netcdf_path)
        self.expert_features = expert_features
        self.gating_features = gating_features
        self.target = target
        self.drop_nan = drop_nan
        
        # Load dataset
        self.ds = xr.open_dataset(self.netcdf_path, engine="netcdf4")
        
        # Flatten spatial dimensions
        self._prepare_flat_data()
        
    def _prepare_flat_data(self):
        """Convert xarray Dataset to flat arrays."""
        # Stack spatial dims
        stacked = self.ds.stack(sample=("time", "lat", "lon"))
        
        # Extract features
        self.X_expert = np.stack(
            [stacked[f].values for f in self.expert_features], 
            axis=1
        )
        self.X_gate = np.stack(
            [stacked[f].values for f in self.gating_features],
            axis=1
        )
        
        # Extract target
        if self.target:
            self.y = stacked[self.target].values
        else:
            self.y = None
        
        # Store coordinates for reconstruction
        self.coords = {
            "lat": stacked["lat"].values,
            "lon": stacked["lon"].values,
            "time": stacked["time"].values,
        }
        
        # Handle NaN values
        if self.drop_nan:
            # Find valid samples (no NaN in features or target)
            mask = ~np.isnan(self.X_expert).any(axis=1)
            mask &= ~np.isnan(self.X_gate).any(axis=1)
            if self.y is not None:
                mask &= ~np.isnan(self.y)
            
            self.X_expert = self.X_expert[mask]
            self.X_gate = self.X_gate[mask]
            if self.y is not None:
                self.y = self.y[mask]
            
            # Update coordinates
            for key in self.coords:
                self.coords[key] = self.coords[key][mask]
            
            self.valid_mask = mask
            self.n_valid = np.sum(mask)
            self.n_total = len(mask)
        else:
            self.valid_mask = np.ones(len(self.X_expert), dtype=bool)
            self.n_valid = len(self.X_expert)
            self.n_total = len(self.X_expert)
    
    def __len__(self) -> int:
        """Number of valid samples."""
        return self.n_valid
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get a single sample.
        
        Returns:
            X_expert: Expert features (D_expert,)
            X_gate: Gating features (D_gate,)
            y: Target value (scalar) or None
        """
        X_expert = torch.from_numpy(self.X_expert[idx]).float()
        X_gate = torch.from_numpy(self.X_gate[idx]).float()
        
        if self.y is not None:
            y = torch.tensor(self.y[idx]).float()
            return X_expert, X_gate, y
        else:
            return X_expert, X_gate, None
    
    def get_dataframe(self) -> pd.DataFrame:
        """Export to pandas DataFrame for symbolic discovery.
        
        Returns:
            DataFrame with all features, target, and coordinates
        """
        data = {}
        
        # Add expert features
        for i, feat in enumerate(self.expert_features):
            data[feat] = self.X_expert[:, i]
        
        # Add gating features (avoid duplicates)
        for i, feat in enumerate(self.gating_features):
            if feat not in data:
                data[feat] = self.X_gate[:, i]
        
        # Add target
        if self.y is not None:
            data[self.target] = self.y
        
        # Add coordinates
        for key, val in self.coords.items():
            data[key] = val
        
        return pd.DataFrame(data)
    
    def close(self):
        """Close underlying NetCDF file."""
        if hasattr(self, 'ds'):
            self.ds.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class RegimeAssignedDataset(Dataset):
    """Dataset with pre-assigned regime labels.
    
    Used for training symbolic experts on regime-specific subsets.
    
    Args:
        base_dataset: ClimateDataset instance
        regime_probs: Regime probability matrix (N, K)
        regime_id: Train only on this regime
        threshold: Minimum probability to include sample
        
    Example:
        >>> base = ClimateDataset("train.nc", ...)
        >>> regime_probs = gating_net(base.X_gate)
        >>> regime_0_data = RegimeAssignedDataset(
        ...     base, regime_probs, regime_id=0, threshold=0.5
        ... )
    """
    
    def __init__(
        self,
        base_dataset: ClimateDataset,
        regime_probs: np.ndarray,
        regime_id: int,
        threshold: float = 0.3,
    ):
        self.base_dataset = base_dataset
        self.regime_id = regime_id
        self.threshold = threshold
        
        # Filter to samples assigned to this regime
        self.mask = regime_probs[:, regime_id] > threshold
        self.indices = np.where(self.mask)[0]
        self.weights = regime_probs[self.mask, regime_id]
        
        # Store subset of data
        self.X_expert = base_dataset.X_expert[self.mask]
        self.y = base_dataset.y[self.mask] if base_dataset.y is not None else None
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get sample with regime weight.
        
        Returns:
            X_expert: Expert features
            y: Target
            weight: Regime probability (for weighted loss)
        """
        X = torch.from_numpy(self.X_expert[idx]).float()
        y = torch.tensor(self.y[idx]).float()
        w = torch.tensor(self.weights[idx]).float()
        return X, y, w


class SpatialBatchSampler:
    """Custom sampler that groups nearby spatial points into batches.
    
    Improves training efficiency by exploiting spatial correlation.
    
    Args:
        dataset: ClimateDataset instance
        batch_size: Number of samples per batch
        shuffle: Shuffle spatial tiles
    """
    
    def __init__(
        self,
        dataset: ClimateDataset,
        batch_size: int = 512,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create spatial tiles
        self._create_tiles()
    
    def _create_tiles(self):
        """Group samples into spatial tiles."""
        # Get unique lat/lon grid points
        lats = np.unique(self.dataset.coords["lat"])
        lons = np.unique(self.dataset.coords["lon"])
        
        # Create tile indices (e.g., 10x10 degree tiles)
        lat_bins = np.linspace(-90, 90, 19)  # 18 tiles
        lon_bins = np.linspace(-180, 180, 37)  # 36 tiles
        
        # Assign each sample to a tile
        lat_idx = np.digitize(self.dataset.coords["lat"], lat_bins)
        lon_idx = np.digitize(self.dataset.coords["lon"], lon_bins)
        tile_id = lat_idx * len(lon_bins) + lon_idx
        
        # Group samples by tile
        self.tiles = {}
        for i, tid in enumerate(tile_id):
            if tid not in self.tiles:
                self.tiles[tid] = []
            self.tiles[tid].append(i)
        
        # Convert to arrays
        self.tile_indices = [np.array(indices) for indices in self.tiles.values()]
    
    def __iter__(self):
        """Generate batches."""
        # Shuffle tiles
        tiles = self.tile_indices.copy()
        if self.shuffle:
            np.random.shuffle(tiles)
        
        # Generate batches from tiles
        for tile in tiles:
            # Shuffle within tile
            if self.shuffle:
                np.random.shuffle(tile)
            
            # Yield batches
            for i in range(0, len(tile), self.batch_size):
                batch = tile[i:i + self.batch_size]
                yield batch.tolist()
    
    def __len__(self) -> int:
        """Total number of batches."""
        return sum(len(tile) // self.batch_size + (1 if len(tile) % self.batch_size else 0)
                   for tile in self.tile_indices)


def create_dataloaders(
    train_path: str | Path,
    test_path: str | Path,
    expert_features: List[str],
    gating_features: List[str],
    target: str,
    batch_size: int = 512,
    num_workers: int = 0,
    use_spatial_batching: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test DataLoaders.
    
    Args:
        train_path: Path to train_dataset.nc
        test_path: Path to test_dataset.nc
        expert_features: Features for experts
        gating_features: Features for gating
        target: Target variable
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_spatial_batching: Use spatial-aware batching
        
    Returns:
        train_loader, test_loader
        
    Example:
        >>> train_loader, test_loader = create_dataloaders(
        ...     "data/processed/train_dataset.nc",
        ...     "data/processed/test_dataset.nc",
        ...     expert_features=["sst", "sss", "log_chl"],
        ...     gating_features=["lat_norm", "lon_norm", "sst"],
        ...     target="fco2",
        ...     batch_size=1024,
        ... )
    """
    # Create datasets
    train_dataset = ClimateDataset(
        train_path,
        expert_features=expert_features,
        gating_features=gating_features,
        target=target,
        drop_nan=True,
    )
    
    test_dataset = ClimateDataset(
        test_path,
        expert_features=expert_features,
        gating_features=gating_features,
        target=target,
        drop_nan=True,
    )
    
    # Create samplers
    if use_spatial_batching:
        train_sampler = SpatialBatchSampler(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, test_loader