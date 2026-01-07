import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class ClimateSpatialDataset(Dataset):
    def __init__(self, nc_path, features, mode='train'):
        """
        Loads climate data as spatial maps (Time, Channels, Lat, Lon).
        """
        super().__init__()
        try:
            self.ds = xr.open_dataset(nc_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find data file at: {nc_path}")
            
        # Time Slicing (Train vs Test)
        if mode == 'train':
            self.ds = self.ds.sel(time=slice(None, "2019-12-31"))
        else:
            self.ds = self.ds.sel(time=slice("2020-01-01", None))
            
        # 1. Prepare Features: (Time, Channels, Lat, Lon)
        try:
            data_xr = self.ds[features].to_array(dim='channel')
            data_xr = data_xr.transpose('time', 'channel', 'lat', 'lon')
            self.data = torch.from_numpy(data_xr.values).float()
        except KeyError:
            raise KeyError(f"Missing features. Available: {list(self.ds.data_vars)}")

        # 2. Create Global Mask (Where is the Ocean?)
        # CRITICAL: Create mask BEFORE normalization to avoid NaN issues
        # Shape: (Time, Lat, Lon) -> Aggregated to (Lat, Lon)
        # We check specific channel 0 (SST) for validity
        valid_mask_time = ~np.isnan(self.data[:, 0, :, :].numpy())
        freq_mask = valid_mask_time.mean(axis=0)
        self.mask = torch.from_numpy(freq_mask > 0.01) # >1% presence

        print(f"Mask active pixels: {self.mask.sum()} (Coverage: {self.mask.float().mean():.4f})")

        # 3. Robust Normalization (Ignore NaNs)
        print("Normalizing data (ignoring NaNs)...")
        # We iterate channels to calculate nanmean/nanstd safely
        means = []
        stds = []
        for c in range(self.data.shape[1]):
            channel_data = self.data[:, c, :, :]
            # Select only valid ocean pixels to calculate stats
            valid_pixels = channel_data[torch.from_numpy(valid_mask_time)]
            
            mu = valid_pixels.mean()
            sigma = valid_pixels.std()
            
            means.append(mu)
            stds.append(sigma)
            
            # Apply Normalization
            self.data[:, c, :, :] = (channel_data - mu) / (sigma + 1e-6)
        
        # Store stats for later (e.g. PySR un-normalization)
        self.X_mean = torch.tensor(means)
        self.X_std = torch.tensor(stds)

        # 4. Handle NaNs (Fill Land with 0.0)
        # Now that we have normalized, 0.0 represents the 'Mean' value, which is safe for ML
        self.data = torch.nan_to_num(self.data, nan=0.0)
        
        # 5. Teacher Targets (Placeholder)
        self.teacher_targets = None

    def set_teacher_targets(self, targets):
        """
        Stores the K-Means labels (The 'Teacher') for training.
        Args:
            targets (Tensor): Shape (Time, Lat, Lon) of integer labels.
        """
        assert targets.shape == (self.data.shape[0], self.data.shape[2], self.data.shape[3])
        self.teacher_targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            'image': self.data[idx],   # (C, H, W)
            'mask': self.mask          # (H, W)
        }
        
        if self.teacher_targets is not None:
            item['target'] = self.teacher_targets[idx] # (H, W)
            
        return item