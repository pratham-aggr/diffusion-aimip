"""
AIMIP Forcing Data Loader

Loads and processes AIMIP monthly forcing data (SST and sea-ice concentration).
Interpolates to model's native grid and 6-hourly timesteps.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import warnings

try:
    import xesmf
    HAS_XESMF = True
except ImportError:
    HAS_XESMF = False
    warnings.warn("xesmf not installed. Regridding will not work. Install with: pip install xesmf")


def load_aimip_forcing_monthly(
    forcing_dir: str,
    year: int,
    month: int,
    variable: str = "sst",  # "sst" or "seaice"
) -> xr.Dataset:
    """
    Load AIMIP monthly forcing file.
    
    Expected filename format:
    - SST: sst_YYYYMM.nc or similar
    - Sea-ice: seaice_YYYYMM.nc or similar
    
    Args:
        forcing_dir: Directory containing AIMIP forcing files
        year: Year
        month: Month (1-12)
        variable: Variable name ("sst" or "seaice")
    
    Returns:
        xarray Dataset with forcing data
    """
    forcing_path = Path(forcing_dir)
    
    # Try different filename patterns
    patterns = [
        f"{variable}_{year:04d}{month:02d}.nc",
        f"{variable}_{year:04d}-{month:02d}.nc",
        f"{variable}_AIMIP_{year:04d}{month:02d}.nc",
    ]
    
    for pattern in patterns:
        filepath = forcing_path / pattern
        if filepath.exists():
            return xr.open_dataset(filepath)
    
    # If not found, try loading from a single file with time dimension
    # (AIMIP might provide all months in one file)
    all_files = list(forcing_path.glob(f"*{variable}*.nc"))
    if all_files:
        ds = xr.open_dataset(all_files[0])
        # Select by time
        target_time = datetime(year, month, 15)  # Mid-month
        if "time" in ds.coords:
            ds = ds.sel(time=target_time, method="nearest")
        return ds
    
    raise FileNotFoundError(
        f"Could not find AIMIP forcing file for {variable} {year}-{month:02d} in {forcing_dir}"
    )


def interpolate_monthly_to_6hourly(
    monthly_data: xr.Dataset,
    start_date: datetime,
    end_date: datetime,
    method: str = "linear",
) -> xr.Dataset:
    """
    Interpolate monthly forcing data to 6-hourly timesteps.
    
    Args:
        monthly_data: Monthly forcing data (xarray Dataset)
        start_date: Start date for 6-hourly output
        end_date: End date for 6-hourly output
        method: Interpolation method ("linear", "nearest", "cubic")
    
    Returns:
        xarray Dataset with 6-hourly timesteps
    """
    # Create 6-hourly time coordinates
    times = []
    current = start_date
    while current <= end_date:
        times.append(current)
        current += timedelta(hours=6)
    
    time_coord = xr.DataArray(times, dims=["time"], coords={"time": times})
    
    # Reindex to 6-hourly and interpolate
    if "time" in monthly_data.coords:
        # Ensure monthly data has proper time coordinate
        monthly_data = monthly_data.swap_dims({"time": "time"}) if "time" in monthly_data.dims else monthly_data
        
        # Reindex and interpolate
        data_6h = monthly_data.reindex(time=times, method=method)
        
        # Forward fill for any remaining NaN values
        data_6h = data_6h.ffill(dim="time").bfill(dim="time")
    else:
        # If no time dimension, broadcast to all timesteps
        data_6h = monthly_data.expand_dims("time").assign_coords(time=times)
        data_6h = data_6h.reindex(time=times, method="ffill")
    
    return data_6h


def regrid_to_model_grid(
    forcing_data: xr.Dataset,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    method: str = "conservative",
) -> xr.Dataset:
    """
    Regrid forcing data to model's native grid.
    
    Args:
        forcing_data: Forcing data (xarray Dataset)
        target_lat: Target latitude coordinates
        target_lon: Target longitude coordinates
        method: Regridding method ("conservative", "bilinear", "nearest_s2d")
    
    Returns:
        Regridded xarray Dataset
    """
    if not HAS_XESMF:
        raise ImportError(
            "xesmf is required for regridding. Install with: pip install xesmf"
        )
    
    # Create target grid
    target_grid = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })
    
    # Create regridder
    regridder = xesmf.Regridder(
        forcing_data,
        target_grid,
        method=method,
        periodic=True,  # Important for global grids
    )
    
    # Regrid all data variables
    regridded = regridder(forcing_data)
    
    # Clean up
    regridder.clean_weight_file()
    
    return regridded


def load_aimip_forcing_full_period(
    forcing_dir: str,
    start_date: datetime,
    end_date: datetime,
    target_lat: Optional[np.ndarray] = None,
    target_lon: Optional[np.ndarray] = None,
    variables: list = ["sst", "seaice"],
) -> xr.Dataset:
    """
    Load and process AIMIP forcing for full simulation period.
    
    Args:
        forcing_dir: Directory containing AIMIP forcing files
        start_date: Start date
        end_date: End date
        target_lat: Target latitude grid (for regridding)
        target_lon: Target longitude grid (for regridding)
        variables: List of variables to load ["sst", "seaice"]
    
    Returns:
        xarray Dataset with 6-hourly forcing data on model grid
    """
    all_data = []
    
    # Load monthly data for each month in the period
    current = start_date.replace(day=1)  # Start of month
    while current <= end_date:
        year = current.year
        month = current.month
        
        month_data_vars = {}
        for var in variables:
            try:
                var_data = load_aimip_forcing_monthly(forcing_dir, year, month, variable=var)
                # Extract the variable (assuming it's named after the variable or has standard name)
                if var in var_data.data_vars:
                    month_data_vars[var] = var_data[var]
                elif len(var_data.data_vars) == 1:
                    # If only one variable, use it
                    month_data_vars[var] = list(var_data.data_vars.values())[0]
                else:
                    # Try common names
                    for name in var_data.data_vars:
                        if var.lower() in name.lower() or "sea_surface_temperature" in name.lower() or "sea_ice" in name.lower():
                            month_data_vars[var] = var_data[name]
                            break
                    else:
                        raise ValueError(f"Could not find {var} in {var_data.data_vars}")
            except FileNotFoundError as e:
                warnings.warn(f"Could not load {var} for {year}-{month:02d}: {e}")
                continue
        
        if month_data_vars:
            # Combine into single dataset
            month_ds = xr.Dataset(month_data_vars)
            month_ds = month_ds.expand_dims("time").assign_coords(
                time=[datetime(year, month, 15)]  # Mid-month
            )
            all_data.append(month_ds)
        
        # Move to next month
        if month == 12:
            current = current.replace(year=year + 1, month=1)
        else:
            current = current.replace(month=month + 1)
    
    if not all_data:
        raise ValueError(f"No forcing data found for period {start_date} to {end_date}")
    
    # Concatenate all months
    full_data = xr.concat(all_data, dim="time")
    
    # Interpolate to 6-hourly
    data_6h = interpolate_monthly_to_6hourly(full_data, start_date, end_date)
    
    # Regrid to model grid if provided
    if target_lat is not None and target_lon is not None:
        data_6h = regrid_to_model_grid(data_6h, target_lat, target_lon)
    
    return data_6h


def get_forcing_at_timestep(
    forcing_data: xr.Dataset,
    timestep: int,
    timestep_delta_hours: int = 6,
) -> Dict[str, np.ndarray]:
    """
    Extract forcing data for a specific timestep.
    
    Args:
        forcing_data: Full 6-hourly forcing dataset
        timestep: Timestep index (0-based)
        timestep_delta_hours: Hours per timestep (default: 6)
    
    Returns:
        Dictionary of {variable_name: array} for the timestep
    """
    if timestep >= len(forcing_data.time):
        # Use last available timestep
        timestep = len(forcing_data.time) - 1
    
    step_data = forcing_data.isel(time=timestep)
    
    result = {}
    for var in step_data.data_vars:
        result[var] = step_data[var].values
    
    return result


