"""
Ensemble generation utilities for AIMIP-1 simulations.

Creates multiple ensemble members with different initial conditions.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from datetime import datetime
import xarray as xr


def create_noise_perturbed_ensemble(
    base_state: Dict[str, torch.Tensor],
    n_members: int = 5,
    noise_scale: float = 0.01,
    seed: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Create ensemble by adding small Gaussian noise to base initial condition.
    
    Args:
        base_state: Dictionary of {variable_name: tensor} initial condition
        n_members: Number of ensemble members
        noise_scale: Standard deviation of noise (relative to variable std)
        seed: Random seed for reproducibility
    
    Returns:
        List of perturbed initial conditions
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    ensemble = []
    for i in range(n_members):
        perturbed_state = {}
        for var_name, tensor in base_state.items():
            # Add noise proportional to variable magnitude
            if isinstance(tensor, torch.Tensor):
                std = tensor.std().item()
                noise = torch.randn_like(tensor) * noise_scale * std
                perturbed_state[var_name] = tensor + noise
            else:
                perturbed_state[var_name] = tensor
        ensemble.append(perturbed_state)
    
    return ensemble


def create_successive_days_ensemble(
    datamodule,
    base_date: datetime,
    n_members: int = 5,
) -> List[Dict[str, torch.Tensor]]:
    """
    Create ensemble using successive days from ERA5 as initial conditions.
    
    Args:
        datamodule: ERA5 datamodule for loading data
        base_date: Base date (first member will be this date)
        n_members: Number of ensemble members
    
    Returns:
        List of initial conditions from successive days
    """
    from datetime import timedelta
    
    ensemble = []
    datamodule.setup("validate")
    
    for i in range(n_members):
        member_date = base_date + timedelta(days=i)
        # TODO: Load ERA5 state at member_date
        # This requires implementing a function to load specific timestep from ERA5
        # For now, placeholder
        raise NotImplementedError(
            "Successive days ensemble requires ERA5 data loading by date. "
            "Implement datamodule method to load specific timestep."
        )


def create_stochastic_ensemble(
    base_state: Dict[str, torch.Tensor],
    n_members: int = 5,
    seeds: Optional[List[int]] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Create ensemble using different random seeds for stochastic model.
    
    For diffusion models, different random seeds will produce different
    stochastic realizations even from the same initial condition.
    
    Args:
        base_state: Base initial condition (same for all members)
        n_members: Number of ensemble members
        seeds: List of random seeds (one per member)
    
    Returns:
        List of initial conditions (same state, different seeds for inference)
    """
    if seeds is None:
        seeds = list(range(1000, 1000 + n_members))
    
    ensemble = []
    for seed in seeds:
        # Store seed with state for use during inference
        member_state = base_state.copy()
        member_state["_random_seed"] = seed
        ensemble.append(member_state)
    
    return ensemble


def load_era5_initial_condition(
    datamodule,
    date: datetime,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Load ERA5 initial condition at a specific date.
    
    Args:
        datamodule: ERA5 datamodule
        date: Date to load
        device: Device to load tensors on
    
    Returns:
        Dictionary of initial condition tensors
    """
    from src.utilities.aimip_era5_initial_condition import load_era5_initial_condition_via_datamodule
    
    return load_era5_initial_condition_via_datamodule(datamodule, date, device=device)


def create_aimip_ensemble(
    datamodule,
    base_date: datetime,
    n_members: int = 5,
    method: str = "noise_perturbation",
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    """
    Create AIMIP ensemble using specified method.
    
    Args:
        datamodule: ERA5 datamodule
        base_date: Base date for initial condition (Oct 1, 1978 00 UTC)
        n_members: Number of ensemble members (default: 5)
        method: Ensemble generation method:
            - "noise_perturbation": Add noise to ERA5 IC
            - "successive_days": Use 5 successive days from ERA5
            - "stochastic": Same IC, different random seeds
        **kwargs: Additional arguments for specific methods
    
    Returns:
        List of ensemble member initial conditions
    """
    # Load base initial condition
    base_ic = load_era5_initial_condition(datamodule, base_date)
    
    if method == "noise_perturbation":
        noise_scale = kwargs.get("noise_scale", 0.01)
        seed = kwargs.get("seed", None)
        return create_noise_perturbed_ensemble(
            base_ic, n_members=n_members, noise_scale=noise_scale, seed=seed
        )
    
    elif method == "successive_days":
        return create_successive_days_ensemble(datamodule, base_date, n_members=n_members)
    
    elif method == "stochastic":
        seeds = kwargs.get("seeds", None)
        return create_stochastic_ensemble(base_ic, n_members=n_members, seeds=seeds)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}. Choose from: noise_perturbation, successive_days, stochastic")


