"""Consolidated Benchmarking Module

This module consolidates:
- models/baselines.py → benchmarks/models.py
- validation/benchmark.py → benchmarks/runner.py

Usage:
    from climate_discovery.benchmarks import (
        LinearBaseline, RFBaseline, XGBBaseline,
        ModelBenchmark, run_all_benchmarks
    )
"""

from .models import (
    LinearBaseline,
    RFBaseline,
    XGBBaseline,
    LatitudeBandLinearRegression,
    KMeansSymbolicRegressor,
)
from .runner import ModelBenchmark, run_all_benchmarks

__all__ = [
    # Baseline models
    "LinearBaseline",
    "RFBaseline", 
    "XGBBaseline",
    "LatitudeBandLinearRegression",
    "KMeansSymbolicRegressor",
    
    # Benchmarking framework
    "ModelBenchmark",
    "run_all_benchmarks",
]
