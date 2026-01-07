import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class ClimateDataset(Dataset):
    def __init__(self, nc_path, features, target='fco2', mode='train', test_year=2020):
        self.features = features
        self.target = target
        
        # Load Data
        ds = xr.open_dataset(nc_path)
        df = ds.to_dataframe().reset_index().dropna(subset=features + [target])
        
        # Split Train/Test
        if mode == 'train':
            df = df[df.year < test_year]
        else:
            df = df[df.year >= test_year]
            
        self.coords = df[['lat', 'lon']].values.astype(np.float32)
        
        # Normalize
        X = df[features].values.astype(np.float32)
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.X = (X - self.X_mean) / (self.X_std + 1e-6)
        
        self.y = df[target].values.astype(np.float32).reshape(-1, 1)
        self.regime_labels = None 

    def set_kmeans_labels(self, labels):
        if len(labels) != len(self.X):
             raise ValueError(f"Label mismatch: Got {len(labels)}, expected {len(self.X)}")
        self.regime_labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            'features': torch.tensor(self.X[idx]),
            'target': torch.tensor(self.y[idx]),
            'coords': torch.tensor(self.coords[idx])
        }
        if self.regime_labels is not None:
            sample['regime'] = self.regime_labels[idx]
        return sample