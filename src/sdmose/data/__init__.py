"""Data loading and processing utilities."""

# Import main dataset class
try:
    from .datasets import ClimateDataset, create_dataloaders
except ImportError:
    pass
