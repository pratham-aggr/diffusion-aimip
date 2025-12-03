"""
CMOR-ization utilities for AIMIP-1 submission.

Converts model outputs to CMIP-compliant NetCDF format with proper naming,
metadata, and CF conventions.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import xarray as xr
from datetime import datetime

from src.utilities.aimip_cmor_mapping import (
    get_cmip_name,
    get_cmip_units,
    get_cmip_long_name,
    extract_pressure_level,
    AIMIP_PRESSURE_LEVELS_PA,
)


def create_cmip_compliant_dataset(
    data: Dict[str, np.ndarray],
    variable_names: list,
    lat: np.ndarray,
    lon: np.ndarray,
    time: np.ndarray,
    model_name: str,
    ensemble_member: str = "r1i1p1f1",
    grid_label: str = "gn",  # "gn" = native grid, "gr" = regridded
    table_id: str = "Amon",  # "Amon" = monthly, "day" = daily
    experiment_id: str = "aimip",
    start_date: str = "197810",
    end_date: str = "202412",
) -> Dict[str, xr.Dataset]:
    """
    Create CMIP-compliant xarray Datasets from model predictions.
    
    Args:
        data: Dictionary of {variable_name: array} predictions
        variable_names: List of internal variable names
        lat: Latitude coordinates
        lon: Longitude coordinates
        time: Time coordinates (datetime64)
        model_name: Model name (e.g., "EmulationSST-SeaIce-EDM-AIMIP")
        ensemble_member: Ensemble member identifier (e.g., "r1i1p1f1")
        grid_label: Grid label ("gn" for native, "gr" for regridded)
        table_id: CMIP table ID ("Amon" for monthly, "day" for daily)
        experiment_id: Experiment ID ("aimip", "aimip+2K", "aimip+4K")
        start_date: Start date string (YYYYMM)
        end_date: End date string (YYYYMM)
    
    Returns:
        Dictionary of {cmip_name: xr.Dataset} for each variable
    """
    datasets = {}
    
    # Group variables by CMIP name (since multiple pressure levels map to same name)
    cmip_groups: Dict[str, list] = {}
    for var_name in variable_names:
        cmip_name = get_cmip_name(var_name)
        if cmip_name:
            if cmip_name not in cmip_groups:
                cmip_groups[cmip_name] = []
            cmip_groups[cmip_name].append(var_name)
    
    # Create dataset for each CMIP variable
    for cmip_name, internal_vars in cmip_groups.items():
        # Check if this is a 3D field (has pressure levels)
        pressure_levels = []
        var_data_list = []
        
        for internal_var in internal_vars:
            level = extract_pressure_level(internal_var)
            if level is not None:
                pressure_levels.append(level * 100)  # Convert hPa to Pa
                if internal_var in data:
                    var_data_list.append((level * 100, data[internal_var]))
            else:
                # Surface field (2D)
                if internal_var in data:
                    var_data = data[internal_var]
                    break
        else:
            # 3D field - stack by pressure level
            if var_data_list:
                # Sort by pressure level (descending)
                var_data_list.sort(key=lambda x: x[0], reverse=True)
                pressure_levels = [p for p, _ in var_data_list]
                var_data = np.stack([arr for _, arr in var_data_list], axis=1)  # Add pressure dimension
            else:
                continue
        
        # Create coordinates
        coords = {
            "time": time,
            "lat": lat,
            "lon": lon,
        }
        
        # Add pressure dimension for 3D fields
        if len(pressure_levels) > 0:
            coords["plev"] = np.array(pressure_levels)
            dims = ["time", "plev", "lat", "lon"]
        else:
            dims = ["time", "lat", "lon"]
        
        # Create DataArray
        da = xr.DataArray(
            var_data,
            dims=dims,
            coords=coords,
            name=cmip_name,
            attrs={
                "standard_name": cmip_name,
                "long_name": get_cmip_long_name(cmip_name) or cmip_name,
                "units": get_cmip_units(cmip_name) or "",
                "cell_methods": "time: mean" if table_id == "Amon" else "time: point",
            }
        )
        
        # Create Dataset
        ds = xr.Dataset({cmip_name: da})
        
        # Add coordinate attributes (CF-compliant)
        ds["time"].attrs = {
            "standard_name": "time",
            "long_name": "time",
            "axis": "T",
            "calendar": "standard",
        }
        ds["lat"].attrs = {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        }
        ds["lon"].attrs = {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        }
        
        if "plev" in ds.coords:
            ds["plev"].attrs = {
                "standard_name": "air_pressure",
                "long_name": "pressure",
                "units": "Pa",
                "axis": "Z",
                "positive": "down",
            }
        
        # Add global attributes (CMIP requirements)
        ds.attrs = {
            "Conventions": "CF-1.7 CMIP-6.2",
            "activity_id": "AIMIP",
            "experiment_id": experiment_id,
            "table_id": table_id,
            "source": f"{model_name} v1.0",
            "institution": "Climate Analytics Lab",  # Update with your institution
            "institution_id": "CAL",  # Update with your ID
            "model_id": model_name,
            "realization_index": ensemble_member,
            "grid_label": grid_label,
            "nominal_resolution": "100 km",  # Approximate for 240x121 grid
            "creation_date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "title": f"{cmip_name} from {model_name}",
            "comment": "AIMIP-1 submission: EDM diffusion model for climate emulation",
        }
        
        datasets[cmip_name] = ds
    
    return datasets


def save_cmip_netcdf(
    datasets: Dict[str, xr.Dataset],
    output_dir: str,
    model_name: str,
    ensemble_member: str = "r1i1p1f1",
    grid_label: str = "gn",
    table_id: str = "Amon",
    experiment_id: str = "aimip",
    start_date: str = "197810",
    end_date: str = "202412",
) -> list:
    """
    Save CMIP-compliant NetCDF files with proper naming convention.
    
    Filename format: CFfieldname_table_MMM_experiment_rXiXpXfX_gX_YYYYMM-YYYYMM.nc
    
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for cmip_name, ds in datasets.items():
        # Create filename
        filename = f"{cmip_name}_{table_id}_{model_name}_{experiment_id}_{ensemble_member}_{grid_label}_{start_date}-{end_date}.nc"
        filepath = output_path / filename
        
        # Save to NetCDF (single precision)
        encoding = {cmip_name: {"dtype": "float32", "zlib": True, "complevel": 4}}
        for coord in ds.coords:
            encoding[coord] = {"dtype": "float32"}
        
        ds.to_netcdf(filepath, encoding=encoding)
        saved_files.append(str(filepath))
        print(f"Saved: {filepath}")
    
    return saved_files
