import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

# Try importing dask for parallel computation
try:
    from dask.distributed import Client, LocalCluster
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

# Suppress dask warnings about divide by zero during std computation
# (these occur when computing std on variables with NaN values, which is expected)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')

# Try importing gcsfs for GCS path support
try:
    import gcsfs
    HAS_GCSFS = True
except ImportError:
    HAS_GCSFS = False

# Configuration
ZARR_PATH = "/data/climate-analytics-lab-shared/ERA5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
# Save to user's home directory since shared directory may be read-only
OUTPUT_DIR = Path.home() / "data" / "ERA5" / "statistics"
TRAIN_PERIOD = slice("1979-01-01", "2020-12-31")  # Standard ERA5 training period

# Variables to exclude from statistics computation (static fields, coordinates)
EXCLUDE_VARS = {
    "latitude", "longitude", "level", "time",
    # Static fields that don't need normalization
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "land_sea_mask",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
}


def compute_statistics(zarr_path: str, output_dir: Path, train_period: slice, use_dask: bool = True):
    """
    Compute mean and standard deviation for ERA5 dataset.
    
    Args:
        zarr_path: Path to the ERA5 zarr dataset
        output_dir: Directory to save statistics files
        train_period: Time slice for computing statistics (typically training period)
        use_dask: Whether to use dask for parallel computation
    """
    # Set up dask client for parallel computation if available
    client = None
    if use_dask and HAS_DASK:
        try:
            # Use threads scheduler by default (better for I/O bound operations)
            # For CPU-bound, could use processes, but threads work better with zarr
            # Use more workers for better parallelization
            n_workers = min(8, (os.cpu_count() or 1) // 2)  # Use half of available cores
            n_workers = max(2, n_workers)  # At least 2 workers
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=2,
                processes=False,  # Use threads for better zarr performance
                silence_logs=50,  # Reduce dask logging
            )
            client = Client(cluster)
            print(f"Using Dask client with {len(cluster.workers)} workers for parallel computation")
        except Exception as e:
            print(f"Warning: Could not set up Dask client ({e}), using default scheduler")
            client = None
    
    print(f"Loading dataset from: {zarr_path}")
    print(f"Computing statistics over period: {train_period.start} to {train_period.stop}")
    
    # Open dataset - handle GCS paths
    # Use optimized chunking: smaller time chunks for better parallelization
    # Based on ERA5 datamodule best practices
    if use_dask and HAS_DASK:
        optimal_chunks = {
            "time": 8,  # Smaller chunks = more parallel work
            "latitude": -1,  # Don't chunk spatial dims (better for reduction ops)
            "longitude": -1,
            "level": -1,
        }
    else:
        optimal_chunks = {"time": 100}  # Larger chunks for non-dask
    
    if zarr_path.startswith("gs://") and HAS_GCSFS:
        # For GCS paths, use gcsfs filesystem
        fs = gcsfs.GCSFileSystem()
        store = fs.get_mapper(zarr_path)
        ds = xr.open_zarr(store, decode_times=True, chunks=optimal_chunks, consolidated=True)
    else:
        # For local paths, use standard open_zarr
        ds = xr.open_zarr(zarr_path, decode_times=True, chunks=optimal_chunks, consolidated=True)
    
    # Convert ERA5 geopotential (m^2/s^2) to height (m) so that statistics
    # and downstream training use zg units expected by CMIP/AIMIP.
    if "geopotential" in ds.data_vars:
        g = 9.80665  # standard gravity [m/s^2]
        ds["geopotential"] = ds["geopotential"] / g

    # Select training period
    ds_train = ds.sel(time=train_period)
    
    print(f"\nDataset info:")
    print(f"  Time steps: {len(ds_train.time)}")
    print(f"  Spatial dims: {ds_train.dims['longitude']}x{ds_train.dims['latitude']}")
    if 'level' in ds_train.dims:
        print(f"  Levels: {ds_train.dims['level']}")
    
    # Get list of variables to process
    all_vars = set(ds_train.data_vars)
    vars_to_process = sorted(all_vars - EXCLUDE_VARS)
    
    print(f"\nProcessing {len(vars_to_process)} variables:")
    print(f"  {', '.join(vars_to_process[:10])}")
    if len(vars_to_process) > 10:
        print(f"  ... and {len(vars_to_process) - 10} more")
    
    # Compute statistics
    print("\nComputing means and standard deviations...")
    means = {}
    stds = {}
    
    # Process variables in parallel batches if using dask
    if use_dask and HAS_DASK and client is not None:
        batch_size = min(4, len(vars_to_process))  # Process 4 variables at a time
        print(f"Processing variables in parallel batches of {batch_size}")
        
        for i in tqdm(range(0, len(vars_to_process), batch_size), desc="Processing variable batches"):
            batch_vars = vars_to_process[i:i+batch_size]
            
            # Compute mean and std for all variables in batch simultaneously
            batch_means = {}
            batch_stds = {}
            
            for var in batch_vars:
                try:
                    data_var = ds_train[var]
                    # Compute both mean and std in parallel (dask will optimize)
                    batch_means[var] = data_var.mean(dim='time')
                    batch_stds[var] = data_var.std(dim='time')
                except Exception as e:
                    print(f"  Error setting up computation for {var}: {e}")
                    continue
            
            # Compute all at once for better parallelization
            if batch_means:
                computed_means = {k: v.compute() for k, v in batch_means.items()}
                computed_stds = {k: v.compute() for k, v in batch_stds.items()}
                
                # Post-process each variable
                for var in batch_means.keys():
                    mean_val = computed_means[var]
                    std_val = computed_stds[var]
                    
                    # Check for NaN or zero std
                    if np.any(np.isnan(std_val.values)):
                        print(f"  Note: {var} has NaN in std (expected for masked variables like SST/sea ice)")
                        std_val = xr.where(np.isnan(std_val), 1.0, std_val)
                        mean_val = xr.where(np.isnan(mean_val), 0.0, mean_val)
                    if np.any(std_val.values == 0):
                        print(f"  Note: {var} has zero std, setting to 1.0")
                        std_val = xr.where(std_val == 0, 1.0, std_val)
                    
                    means[var] = mean_val
                    stds[var] = std_val
    else:
        # Sequential processing (original approach)
        for var in tqdm(vars_to_process, desc="Processing variables"):
            try:
                # Compute mean and std over all dimensions except spatial
                data_var = ds_train[var]
                
                # Compute mean
                mean_val = data_var.mean(dim='time').compute()
                means[var] = mean_val
                
                # Compute std
                std_val = data_var.std(dim='time').compute()
                stds[var] = std_val
                
                # Check for NaN or zero std (normal for variables with spatial masks)
                if np.any(np.isnan(std_val.values)):
                    print(f"  Note: {var} has NaN in std (expected for masked variables like SST/sea ice)")
                    # Replace NaN with 1.0 to avoid normalization issues
                    std_val = xr.where(np.isnan(std_val), 1.0, std_val)
                    mean_val = xr.where(np.isnan(mean_val), 0.0, mean_val)
                    means[var] = mean_val
                    stds[var] = std_val
                if np.any(std_val.values == 0):
                    print(f"  Note: {var} has zero std, setting to 1.0")
                    stds[var] = xr.where(std_val == 0, 1.0, std_val)
                    
            except Exception as e:
                print(f"  Error processing {var}: {e}")
                continue
    
    # Create output datasets
    print("\nCreating output datasets...")
    ds_mean = xr.Dataset(means)
    ds_std = xr.Dataset(stds)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to netCDF
    mean_path = output_dir / "era5_mean.nc"
    std_path = output_dir / "era5_std.nc"
    
    print(f"\nSaving statistics to:")
    print(f"  Mean: {mean_path}")
    print(f"  Std:  {std_path}")
    
    ds_mean.to_netcdf(mean_path)
    ds_std.to_netcdf(std_path)
    
    # Print some statistics
    print("\n" + "="*60)
    print("Statistics computed successfully!")
    print("="*60)
    print(f"\nSample statistics:")
    for var in list(vars_to_process)[:5]:
        if var in means:
            print(f"\n{var}:")
            print(f"  Mean: {float(means[var].mean().values):.4f}")
            print(f"  Std:  {float(stds[var].mean().values):.4f}")
            print(f"  Shape: {means[var].shape}")
    
    print(f"\n✓ Statistics files created in: {output_dir}")
    print(f"\nYou can now run training with:")
    print(f"  python run.py experiment=era5_genie")
    
    # Close dask client if we created one
    if client is not None:
        client.close()
        print("Closed Dask client")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ERA5 normalization statistics.")
    parser.add_argument(
        "--zarr-path",
        default=ZARR_PATH,
        help="Path to ERA5 zarr dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory to write era5_mean.nc and era5_std.nc (default: %(default)s)",
    )
    parser.add_argument(
        "--train-start",
        default=TRAIN_PERIOD.start,
        help="Start date for training slice (default: %(default)s)",
    )
    parser.add_argument(
        "--train-end",
        default=TRAIN_PERIOD.stop,
        help="End date for training slice (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing statistics without prompting.",
    )
    parser.add_argument(
        "--no-dask",
        action="store_true",
        help="Disable dask parallel computation (use default scheduler).",
    )
    return parser.parse_args()


def path_exists(path: str) -> bool:
    """Check if a path exists, handling both local and GCS paths."""
    if path.startswith("gs://"):
        if not HAS_GCSFS:
            print("Warning: gcsfs not installed, cannot verify GCS path. Continuing anyway...")
            return True  # Assume it exists and let xarray handle the error
        try:
            fs = gcsfs.GCSFileSystem()
            return fs.exists(path)
        except Exception as e:
            print(f"Warning: Could not verify GCS path existence: {e}")
            return True  # Assume it exists and let xarray handle the error
    else:
        return os.path.exists(path)


if __name__ == "__main__":
    args = parse_args()
    zarr_path = args.zarr_path
    output_dir = Path(args.output_dir).expanduser()
    train_period = slice(args.train_start, args.train_end)

    # For GCS paths, skip the existence check and let xarray handle it
    # (it will provide better error messages if the path is invalid)
    if not zarr_path.startswith("gs://") and not path_exists(zarr_path):
        print(f"Error: Dataset not found at {zarr_path}")
        exit(1)

    mean_file = output_dir / "era5_mean.nc"
    std_file = output_dir / "era5_std.nc"
    if mean_file.exists() and std_file.exists() and not args.force:
        print(f"Statistics already exist at {output_dir}. Use --force to overwrite.")
        exit(0)

    compute_statistics(zarr_path, output_dir, train_period, use_dask=not args.no_dask)

