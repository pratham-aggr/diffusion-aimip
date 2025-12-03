"""
Load ERA5 initial conditions for AIMIP ensemble generation.

Loads specific timesteps from ERA5 dataset for creating ensemble initial conditions.
"""

import os
import numpy as np
import torch
import xarray as xr
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings


def find_timestep_index_in_era5(
    zarr_path: str,
    target_date: datetime,
    hourly_resolution: int = 6,
) -> Optional[int]:
    """
    Find the timestep index in ERA5 zarr for a given date.
    
    Args:
        zarr_path: Path to ERA5 zarr dataset
        target_date: Target date to find
        hourly_resolution: Hourly resolution of data (default: 6 for 6-hourly)
    
    Returns:
        Timestep index if found, None otherwise
    """
    # Open ERA5 zarr
    ds = xr.open_zarr(zarr_path)
    
    # Convert target_date to numpy datetime64
    target_dt64 = np.datetime64(target_date)
    
    # Find closest timestep
    time_coords = ds.time.values
    time_diffs = np.abs(time_coords - target_dt64)
    closest_idx = np.argmin(time_diffs)
    
    # Check if close enough (within 3 hours)
    closest_time = time_coords[closest_idx]
    time_diff = np.abs((closest_time - target_dt64) / np.timedelta64(1, 'h'))
    
    if time_diff > 3:
        warnings.warn(
            f"Target date {target_date} not found in ERA5. "
            f"Closest timestep is {closest_time} ({time_diff:.1f} hours away)"
        )
        return None
    
    return int(closest_idx)


def load_era5_state_at_date(
    zarr_path: str,
    date: datetime,
    input_vars: list,
    output_vars: list,
    normalizer=None,
    in_packer=None,
    out_packer=None,
    hourly_resolution: int = 6,
) -> Dict[str, torch.Tensor]:
    """
    Load ERA5 state (inputs and outputs) at a specific date.
    
    Args:
        zarr_path: Path to ERA5 zarr dataset
        date: Date to load
        input_vars: List of input variable names
        output_vars: List of output variable names
        normalizer: Normalizer object (optional, for normalization)
        in_packer: Packer for input variables (optional)
        out_packer: Packer for output variables (optional)
        hourly_resolution: Hourly resolution (default: 6)
    
    Returns:
        Dictionary with 'inputs' and 'outputs' tensors
    """
    # Find timestep index
    timestep_idx = find_timestep_index_in_era5(zarr_path, date, hourly_resolution)
    
    if timestep_idx is None:
        raise ValueError(f"Could not find timestep for date {date}")
    
    # Open ERA5 zarr
    ds = xr.open_zarr(zarr_path)
    
    # Load input variables
    input_data = {}
    for var in input_vars:
        if var in ds.data_vars:
            var_data = ds[var].isel(time=timestep_idx)
            input_data[var] = var_data.values
        else:
            warnings.warn(f"Input variable {var} not found in ERA5")
    
    # Load output variables
    output_data = {}
    for var in output_vars:
        # Handle 3D variables with pressure levels
        if "_" in var and var.split("_")[-1].isdigit():
            # 3D variable at specific level
            base_name = "_".join(var.split("_")[:-1])
            level = int(var.split("_")[-1])
            
            if base_name in ds.data_vars:
                var_data = ds[base_name].isel(time=timestep_idx)
                # Select level if it exists
                if "level" in var_data.dims:
                    # Find closest level
                    levels = var_data.level.values
                    level_idx = np.argmin(np.abs(levels - level))
                    var_data = var_data.isel(level=level_idx)
                output_data[var] = var_data.values
        else:
            # 2D variable
            if var in ds.data_vars:
                var_data = ds[var].isel(time=timestep_idx)
                output_data[var] = var_data.values
            else:
                warnings.warn(f"Output variable {var} not found in ERA5")
    
    # Convert to tensors
    input_tensors = {}
    output_tensors = {}
    
    for var, data in input_data.items():
        if isinstance(data, np.ndarray):
            # Add batch dimension
            tensor = torch.from_numpy(data).float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)  # Add channel dimension if 2D
            input_tensors[var] = tensor
    
    for var, data in output_data.items():
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)  # Add channel dimension if 2D
            output_tensors[var] = tensor
    
    # Normalize if normalizer provided
    if normalizer is not None:
        for var in input_tensors:
            if var in normalizer.mean and var in normalizer.std:
                mean = normalizer.mean[var].values
                std = normalizer.std[var].values
                input_tensors[var] = (input_tensors[var] - torch.from_numpy(mean).float()) / torch.from_numpy(std).float()
        
        for var in output_tensors:
            if var in normalizer.mean and var in normalizer.std:
                mean = normalizer.mean[var].values
                std = normalizer.std[var].values
                output_tensors[var] = (output_tensors[var] - torch.from_numpy(mean).float()) / torch.from_numpy(std).float()
    
    # Pack if packers provided
    if in_packer is not None:
        inputs_packed = in_packer.pack(input_tensors, axis=1)
    else:
        # Stack manually if no packer
        inputs_packed = torch.stack(list(input_tensors.values()), dim=1)
    
    if out_packer is not None:
        outputs_packed = out_packer.pack(output_tensors, axis=1)
    else:
        outputs_packed = torch.stack(list(output_tensors.values()), dim=1)
    
    return {
        "inputs": inputs_packed,
        "outputs": outputs_packed,
        "inputs_dict": input_tensors,
        "outputs_dict": output_tensors,
    }


def load_era5_initial_condition_via_datamodule(
    datamodule,
    date: datetime,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load ERA5 initial condition using datamodule's infrastructure.
    
    This is the preferred method as it uses the datamodule's normalization
    and packing logic.
    
    Args:
        datamodule: ERA5 datamodule (must be setup)
        date: Date to load
        device: Device to load tensors on
    
    Returns:
        Dictionary of initial condition tensors ready for model
    """
    # Setup datamodule if not already done
    if not hasattr(datamodule, "train_dataset") or datamodule.train_dataset is None:
        datamodule.setup("validate")
    
    # Get zarr path from datamodule
    zarr_path = datamodule.dataset_path
    
    # Get input and output variables
    input_vars = datamodule.input_vars if hasattr(datamodule, "input_vars") else []
    output_vars = datamodule.output_vars if hasattr(datamodule, "output_vars") else []
    
    # Get normalizer and packers
    normalizer = datamodule.normalizer if hasattr(datamodule, "normalizer") else None
    in_packer = datamodule.in_only_packer if hasattr(datamodule, "in_only_packer") else None
    out_packer = datamodule.out_packer if hasattr(datamodule, "out_packer") else None
    
    # Get hourly resolution
    hourly_resolution = getattr(datamodule.hparams, "hourly_resolution", 6)
    
    # Load state
    state = load_era5_state_at_date(
        zarr_path=zarr_path,
        date=date,
        input_vars=input_vars,
        output_vars=output_vars,
        normalizer=normalizer,
        in_packer=in_packer,
        out_packer=out_packer,
        hourly_resolution=hourly_resolution,
    )
    
    # Move to device
    state["inputs"] = state["inputs"].to(device)
    state["outputs"] = state["outputs"].to(device)
    
    return state


