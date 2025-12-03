"""
AIMIP Forcing Data Loader using ERA5

Loads SST and sea-ice forcing directly from ERA5 zarr dataset.
No need for separate AIMIP forcing dataset - ERA5 has everything we need!
"""

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import warnings


def load_era5_forcing_from_zarr(
    zarr_path: str,
    start_date: datetime,
    end_date: datetime,
    variables: list = ["sea_surface_temperature", "sea_ice_cover"],
    hourly_resolution: int = 6,
) -> xr.Dataset:
    """
    Load SST and sea-ice forcing directly from ERA5 zarr.
    
    Args:
        zarr_path: Path to ERA5 zarr dataset
        start_date: Start date for forcing
        end_date: End date for forcing
        variables: List of variables to load (default: SST and sea-ice)
        hourly_resolution: Hourly resolution of data (default: 6 for 6-hourly)
    
    Returns:
        xarray Dataset with forcing data at specified resolution
    """
    print(f"Loading ERA5 forcing from {zarr_path}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Variables: {variables}")
    
    # Open ERA5 zarr
    ds = xr.open_zarr(zarr_path)
    
    # Select time slice
    time_slice = slice(start_date, end_date)
    
    # Select variables
    forcing_vars = {}
    for var in variables:
        if var in ds.data_vars:
            forcing_vars[var] = ds[var]
        else:
            warnings.warn(f"Variable {var} not found in ERA5 dataset. Available: {list(ds.data_vars.keys())}")
    
    if not forcing_vars:
        raise ValueError(f"None of the requested variables {variables} found in ERA5 dataset")
    
    # Create dataset with selected variables
    forcing_ds = xr.Dataset(forcing_vars)
    
    # Select time slice
    forcing_ds = forcing_ds.sel(time=time_slice)
    
    # If data is at different resolution, resample to desired resolution
    if hourly_resolution > 1:
        # Resample to 6-hourly (or other resolution)
        # ERA5 might be hourly, so we need to resample
        time_freq = f"{hourly_resolution}H"
        forcing_ds = forcing_ds.resample(time=time_freq).mean()
    
    print(f"Loaded forcing data: {len(forcing_ds.time)} timesteps")
    print(f"Time range: {forcing_ds.time.min().values} to {forcing_ds.time.max().values}")
    
    return forcing_ds


def get_era5_forcing_at_timestep(
    forcing_data: xr.Dataset,
    timestep: int,
    variables: list = ["sea_surface_temperature", "sea_ice_cover"],
) -> Dict[str, np.ndarray]:
    """
    Extract forcing data for a specific timestep from ERA5 dataset.
    
    Args:
        forcing_data: ERA5 forcing dataset
        timestep: Timestep index (0-based)
        variables: Variables to extract
    
    Returns:
        Dictionary of {variable_name: array} for the timestep
    """
    if timestep >= len(forcing_data.time):
        # Use last available timestep
        timestep = len(forcing_data.time) - 1
        warnings.warn(f"Timestep {timestep} out of range, using last timestep")
    
    step_data = forcing_data.isel(time=timestep)
    
    result = {}
    for var in variables:
        if var in step_data.data_vars:
            result[var] = step_data[var].values
        elif var in step_data.coords:
            result[var] = step_data[var].values
        else:
            # Try to find similar variable name
            for key in step_data.data_vars.keys():
                if var.lower() in key.lower() or key.lower() in var.lower():
                    result[var] = step_data[key].values
                    break
            else:
                warnings.warn(f"Variable {var} not found at timestep {timestep}")
    
    return result


def load_aimip_forcing_full_period_era5(
    zarr_path: str,
    start_date: datetime,
    end_date: datetime,
    target_lat: Optional[np.ndarray] = None,
    target_lon: Optional[np.ndarray] = None,
    hourly_resolution: int = 6,
) -> xr.Dataset:
    """
    Load AIMIP forcing for full simulation period from ERA5.
    
    This is the main function to use - it loads SST and sea-ice directly from ERA5.
    No separate AIMIP forcing dataset needed!
    
    Args:
        zarr_path: Path to ERA5 zarr dataset
        start_date: Start date (Oct 1, 1978 for AIMIP)
        end_date: End date (Jan 1, 2025 for AIMIP)
        target_lat: Target latitude grid (for regridding, optional)
        target_lon: Target longitude grid (for regridding, optional)
        hourly_resolution: Hourly resolution (default: 6 for 6-hourly)
    
    Returns:
        xarray Dataset with 6-hourly forcing data
    """
    # Load directly from ERA5
    forcing_data = load_era5_forcing_from_zarr(
        zarr_path=zarr_path,
        start_date=start_date,
        end_date=end_date,
        variables=["sea_surface_temperature", "sea_ice_cover"],
        hourly_resolution=hourly_resolution,
    )
    
    # Regrid if target grid provided (usually not needed if using same grid)
    if target_lat is not None and target_lon is not None:
        try:
            import xesmf
            # Check if regridding is needed
            current_lat = forcing_data.latitude.values
            current_lon = forcing_data.longitude.values
            
            if not np.allclose(current_lat, target_lat) or not np.allclose(current_lon, target_lon):
                print("Regridding forcing data to target grid...")
                target_grid = xr.Dataset({
                    "lat": (["lat"], target_lat),
                    "lon": (["lon"], target_lon),
                })
                
                regridder = xesmf.Regridder(
                    forcing_data,
                    target_grid,
                    method="conservative",
                    periodic=True,
                )
                
                forcing_data = regridder(forcing_data)
                regridder.clean_weight_file()
                print("Regridding complete")
            else:
                print("Forcing data already on target grid, skipping regridding")
        except ImportError:
            warnings.warn("xesmf not installed. Skipping regridding. Install with: pip install xesmf")
    
    return forcing_data


