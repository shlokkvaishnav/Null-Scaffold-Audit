"""Spatial and table datasets, plus load_table_data helper."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

try:
    from climate_discovery.config import TEST_START, TRAIN_END
except Exception:
    TRAIN_END = "2019-12-31"
    TEST_START = "2020-01-01"


class ClimateSpatialDataset(Dataset):
    """Loads climate data as spatial maps (Time, Channels, Lat, Lon)."""

    def __init__(
        self,
        nc_path: str,
        features: List[str],
        mode: str = "train",
        train_end: Optional[str] = None,
        test_start: Optional[str] = None,
    ):
        super().__init__()
        nc_path = Path(nc_path)
        if not nc_path.exists():
            raise FileNotFoundError(f"Data file not found: {nc_path}")
        self.ds = xr.open_dataset(nc_path)

        train_end = train_end or TRAIN_END
        test_start = test_start or TEST_START
        tdim = "time" if "time" in self.ds.dims else "tmnth"

        if mode == "train":
            self.ds = self.ds.sel({tdim: slice(None, train_end)})
        else:
            self.ds = self.ds.sel({tdim: slice(test_start, None)})
        self._tdim = tdim

        try:
            data_xr = self.ds[features].to_array(dim="channel")
            data_xr = data_xr.transpose(tdim, "channel", "lat", "lon")
            self.data = torch.from_numpy(data_xr.values).float()
        except KeyError:
            raise KeyError(f"Missing features. Available: {list(self.ds.data_vars)}")

        valid_mask_time = ~np.isnan(self.data[:, 0, :, :].numpy())
        freq_mask = valid_mask_time.mean(axis=0)
        self.mask = torch.from_numpy(freq_mask > 0.01)

        means, stds = [], []
        for c in range(self.data.shape[1]):
            ch = self.data[:, c, :, :]
            valid = ch[torch.from_numpy(valid_mask_time)]
            mu, sig = valid.mean().item(), valid.std().item()
            means.append(mu)
            stds.append(sig)
            self.data[:, c, :, :] = (ch - mu) / (sig + 1e-6)

        self.X_mean = torch.tensor(means)
        self.X_std = torch.tensor(stds)
        self.data = torch.nan_to_num(self.data, nan=0.0)
        self.teacher_targets = None

    def set_teacher_targets(self, targets):
        assert targets.shape == (
            self.data.shape[0],
            self.data.shape[2],
            self.data.shape[3],
        )
        self.teacher_targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {"image": self.data[idx], "mask": self.mask}
        if self.teacher_targets is not None:
            item["target"] = self.teacher_targets[idx]
        return item


def load_table_data(
    train_nc: str,
    test_nc: str,
    feature_cols: List[str],
    target_col: str = "fco2",
    dropna: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Load train/test NetCDF, flatten to (N, F). Returns X_tr, y_tr, lat_tr, year_tr, X_te, y_te, lat_te, year_te.
    """
    import xarray as xr

    def _to_table(path: str):
        ds = xr.open_dataset(path)
        df = ds.to_dataframe().reset_index()
        if "year_feature" not in df.columns and "time" in df.columns:
            ymin, ymax = int(df["time"].dt.year.min()), int(df["time"].dt.year.max())
            df["year_feature"] = (df["time"].dt.year - ymin) / max(ymax - ymin, 1)
        if "lat_norm" not in df.columns and "lat" in df.columns:
            df["lat_norm"] = df["lat"] / 90.0
        if "lon_norm" not in df.columns and "lon" in df.columns:
            df["lon_norm"] = df["lon"] / 180.0
        cols = [c for c in feature_cols if c in df.columns]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        if dropna:
            df = df.dropna(subset=cols + [target_col])
        X = df[cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)
        lat = df["lat"].values
        year = (
            df["time"].dt.year.values
            if "time" in df.columns
            else np.full(len(df), np.nan)
        )
        return X, y, lat, year

    a = _to_table(train_nc)
    b = _to_table(test_nc)
    return (*a, *b)


class ClimateTableDataset:
    """Flattened (N, F) table for baselines and evaluation."""

    def __init__(
        self,
        nc_path: str,
        feature_cols: List[str],
        target_col: str = "fco2",
        mode: str = "train",
        train_nc: Optional[str] = None,
        test_nc: Optional[str] = None,
    ):
        import xarray as xr

        path = (
            Path(train_nc or nc_path) if mode == "train" else Path(test_nc or nc_path)
        )
        ds = xr.open_dataset(path)
        df = ds.to_dataframe().reset_index()
        if "lat_norm" not in df.columns and "lat" in df.columns:
            df["lat_norm"] = df["lat"] / 90.0
        if "lon_norm" not in df.columns and "lon" in df.columns:
            df["lon_norm"] = df["lon"] / 180.0
        df = df.dropna(subset=feature_cols + [target_col])
        self.X = np.asarray(df[feature_cols].values, dtype=np.float32)
        self.y = np.asarray(df[target_col].values, dtype=np.float32)
        self.lat = df["lat"].values
        self.df = df

    @property
    def n_samples(self) -> int:
        return len(self.y)
