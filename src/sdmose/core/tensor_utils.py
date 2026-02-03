"""Core utilities: PyTorch tensor preparation.

Extracted from utils.py for better organization.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def prepare_tensors(
    df: pd.DataFrame,
    expert_cols: List[str],
    gating_cols: List[str],
    target_col: Optional[str] = None,
    drop_nan: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[pd.Index]]:
    """Prepare PyTorch tensors for SD-MoSE training.
    
    Args:
        df: Input DataFrame with all features
        expert_cols: Features for symbolic experts
        gating_cols: Features for gating network
        target_col: Target variable (fCO₂)
        drop_nan: Whether to remove rows with NaN values
        
    Returns:
        X_expert: Expert input tensor (N, D_expert)
        X_gate: Gating input tensor (N, D_gate)
        y: Target tensor (N,) or None
        valid_idx: Index of valid rows (for spatial reconstruction)
        
    Example:
        >>> X_expert, X_gate, y, idx = prepare_tensors(
        ...     df, 
        ...     expert_cols=['sst', 'sss', 'log_chl'],
        ...     gating_cols=['lat_norm', 'lon_norm', 'sst'],
        ...     target_col='fco2'
        ... )
    """
    # Validate columns
    all_cols = set(expert_cols + gating_cols)
    if target_col:
        all_cols.add(target_col)
    
    missing = all_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")
    
    # Handle NaN values
    if drop_nan:
        subset = list(all_cols)
        df_clean = df.dropna(subset=subset)
        valid_idx = df_clean.index
    else:
        df_clean = df
        valid_idx = df.index
    
    if len(df_clean) == 0:
        raise ValueError("No valid data after NaN removal")
    
    # Convert to tensors
    X_expert = torch.tensor(
        df_clean[expert_cols].values, 
        dtype=torch.float32
    )
    X_gate = torch.tensor(
        df_clean[gating_cols].values, 
        dtype=torch.float32
    )
    
    y = None
    if target_col:
        y = torch.tensor(
            df_clean[target_col].values, 
            dtype=torch.float32
        )
    
    return X_expert, X_gate, y, valid_idx
