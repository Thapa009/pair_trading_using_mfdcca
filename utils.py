"""
Unified GPU utility functions - replaces scattered duplicates
"""

import torch
import numpy as np
import pandas as pd
from config import DEVICE


def to_gpu_tensor(data, dtype=torch.float32):
    """
    Single source of truth for GPU conversion
    Replaces: safe_to_gpu, ensure_gpu_tensor (x2)
    """
    if isinstance(data, torch.Tensor):
        return data.to(DEVICE, dtype=dtype)
    elif isinstance(data, pd.Series):
        return torch.tensor(data.values, device=DEVICE, dtype=dtype)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, device=DEVICE, dtype=dtype)
    else:
        return torch.tensor(data, device=DEVICE, dtype=dtype)


import torch


def safe_to_gpu(data, device=None, dtype=None):
    """
    Safely move data to GPU if available.

    Args:
        data: Data to move to GPU (can be torch.Tensor, list, numpy array, etc.)
        device: Specific device to use. If None, uses default GPU if available.
        dtype: Data type for tensor conversion (e.g., torch.float32)

    Returns:
        Data on GPU (or original data if GPU not available)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        if isinstance(data, torch.Tensor):
            tensor_data = data.to(device)
        else:
            # Convert to tensor
            try:
                tensor_data = torch.tensor(
                    data, dtype=dtype if dtype else torch.float32
                ).to(device)
            except:
                # If conversion fails, return original data
                return data

        # Apply dtype if specified
        if dtype is not None:
            tensor_data = tensor_data.to(dtype=dtype)

        return tensor_data
    else:
        # On CPU
        if isinstance(data, torch.Tensor):
            if dtype is not None:
                return data.to(dtype=dtype)
            return data
        else:
            try:
                return torch.tensor(data, dtype=dtype if dtype else torch.float32)
            except:
                return data
