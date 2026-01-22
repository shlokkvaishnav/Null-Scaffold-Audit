"""Data preparation scripts."""

from .download_data import (
    check_copernicus_auth,
    download_chlorophyll,
    download_socat,
)
from .download_data import main as download_main
from .preprocess_data import (
    SCALER_OUTPUT_PATH,
    TEST_OUTPUT_PATH,
    TRAIN_OUTPUT_PATH,
    preprocess,
)

__all__ = [
    "check_copernicus_auth",
    "download_chlorophyll",
    "download_main",
    "download_socat",
    "preprocess",
    "SCALER_OUTPUT_PATH",
    "TEST_OUTPUT_PATH",
    "TRAIN_OUTPUT_PATH",
]
