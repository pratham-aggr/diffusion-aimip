"""
Climate-Specific Metrics and Visualizations for W&B

Custom metrics tailored for climate/weather emulation tasks:
- Spatial skill scores
- Pattern correlation
- Anomaly correlation coefficient (ACC)
- Regional error analysis
- Spectral analysis
- Physical consistency checks
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import wandb


def compute_pattern_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    area_weights: Optional[torch.Tensor] = None
) -> float:
    """
    Compute pattern correlation between predicted and target fields.
    
    Args:
        pred: Predicted field [B, H, W] or [H, W]
        target: Target field [B, H, W] or [H, W]
        area_weights: Optional area weights for latitude weighting [H, W]
    
    Returns:
        Pattern correlation coefficient
    """
    if len(pred.shape) == 3:
        pred = pred.mean(dim=0)  # Average over batch
        target = target.mean(dim=0)
    
    if area_weights is not None:
        # Apply area weighting
        pred_weighted = pred * area_weights
        target_weighted = target * area_weights
    else:
        pred_weighted = pred
        target_weighted = target
    
    # Flatten
    pred_flat = pred_weighted.flatten()
    target_flat = target_weighted.flatten()
    
    # Compute correlation
    corr = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
    
    return float(corr)


def compute_spatial_rmse_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 10
) -> torch.Tensor:
    """
    Compute RMSE in sliding windows across the spatial domain.
    Helps identify regions with high/low errors.
    
    Args:
        pred: Predictions [B, H, W]
        target: Targets [B, H, W]
        window_size: Size of sliding window
    
    Returns:
        RMSE map [H, W]
    """
    # Compute squared errors
    se = (pred - target) ** 2  # [B, H, W]
    mse = se.mean(dim=0)  # Average over batch [H, W]
    
    # Apply moving average filter for smoothing
    kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
    kernel = kernel.to(mse.device)
    
    # Add channel and batch dims for conv2d
    mse_4d = mse.unsqueeze(0).unsqueeze(0)
    
    # Smooth with convolution
    padding = window_size // 2
    mse_smooth = torch.nn.functional.conv2d(mse_4d, kernel, padding=padding)
    
    # Take sqrt to get RMSE
    rmse_map = torch.sqrt(mse_smooth.squeeze())
    
    return rmse_map


def create_error_analysis_plots(
    pred: torch.Tensor,
    target: torch.Tensor,
    var_name: str,
    epoch: int,
) -> Dict[str, wandb.Image]:
    """
    Create comprehensive error analysis plots.
    
    Returns dict of plots:
    - Error histogram
    - QQ plot
    - Residual plot
    - Regional error map
    """
    plots = {}
    
    # Compute errors
    error = (pred - target).cpu().numpy()
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # 1. Error Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(error.flatten(), bins=50, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Prediction Error', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title(f'{var_name}: Error Distribution (Epoch {epoch})', 
                 fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: {error.mean():.2e}\nStd: {error.std():.2e}\nSkew: {_skewness(error):.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plots[f"analysis/error_histogram/{var_name}"] = wandb.Image(fig)
    plt.close(fig)
    
    # 2. QQ Plot (check if errors are normally distributed)
    fig, ax = plt.subplots(figsize=(6, 6))
    error_sorted = np.sort(error.flatten())
    n = len(error_sorted)
    theoretical_quantiles = np.random.normal(error.mean(), error.std(), n)
    theoretical_quantiles.sort()
    
    ax.scatter(theoretical_quantiles[::max(1, n//1000)], 
              error_sorted[::max(1, n//1000)], 
              alpha=0.5, s=2)
    
    # Add 1:1 line
    min_val = min(theoretical_quantiles.min(), error_sorted.min())
    max_val = max(theoretical_quantiles.max(), error_sorted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel('Theoretical Quantiles', fontweight='bold')
    ax.set_ylabel('Sample Quantiles', fontweight='bold')
    ax.set_title(f'{var_name}: Q-Q Plot (Epoch {epoch})', 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plots[f"analysis/qq_plot/{var_name}"] = wandb.Image(fig)
    plt.close(fig)
    
    # 3. Residual vs Predicted (check for heteroscedasticity)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Subsample for visualization
    n_points = pred_np.size
    if n_points > 10000:
        indices = np.random.choice(n_points, 10000, replace=False)
        pred_plot = pred_np.flatten()[indices]
        error_plot = error.flatten()[indices]
    else:
        pred_plot = pred_np.flatten()
        error_plot = error.flatten()
    
    ax.hexbin(pred_plot, error_plot, gridsize=50, cmap='YlOrRd', mincnt=1)
    ax.axhline(0, color='blue', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Predicted Value', fontweight='bold')
    ax.set_ylabel('Residual (Error)', fontweight='bold')
    ax.set_title(f'{var_name}: Residual Plot (Epoch {epoch})', 
                fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plots[f"analysis/residual_plot/{var_name}"] = wandb.Image(fig)
    plt.close(fig)
    
    return plots


def create_regional_analysis(
    pred: torch.Tensor,
    target: torch.Tensor,
    var_name: str,
    epoch: int,
) -> Dict[str, wandb.Image]:
    """
    Analyze model performance by region (tropics, mid-latitudes, poles).
    """
    plots = {}
    
    # Assuming spatial dimensions are [lat, lon]
    n_lat = pred.shape[-2]
    
    # Define regions (approximate)
    tropics = slice(n_lat//3, 2*n_lat//3)
    northern = slice(0, n_lat//3)
    southern = slice(2*n_lat//3, n_lat)
    
    regions = {
        'Tropics': tropics,
        'Northern': northern,
        'Southern': southern,
    }
    
    # Compute RMSE by region
    regional_rmse = {}
    regional_corr = {}
    
    for region_name, region_slice in regions.items():
        pred_region = pred[..., region_slice, :]
        target_region = target[..., region_slice, :]
        
        # RMSE
        rmse = torch.sqrt(((pred_region - target_region) ** 2).mean()).item()
        regional_rmse[region_name] = rmse
        
        # Correlation
        pred_flat = pred_region.flatten()
        target_flat = target_region.flatten()
        corr = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1].item()
        regional_corr[region_name] = corr
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMSE by region
    regions_list = list(regional_rmse.keys())
    rmse_values = list(regional_rmse.values())
    ax1.bar(regions_list, rmse_values, color='coral', edgecolor='black')
    ax1.set_ylabel('RMSE', fontweight='bold')
    ax1.set_title(f'{var_name}: RMSE by Region', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, v in enumerate(rmse_values):
        ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Correlation by region
    corr_values = list(regional_corr.values())
    ax2.bar(regions_list, corr_values, color='skyblue', edgecolor='black')
    ax2.set_ylabel('Correlation', fontweight='bold')
    ax2.set_title(f'{var_name}: Correlation by Region', fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(0.9, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target (0.9)')
    ax2.legend()
    
    # Add values on bars
    for i, v in enumerate(corr_values):
        ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'Regional Performance Analysis (Epoch {epoch})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plots[f"analysis/regional_performance/{var_name}"] = wandb.Image(fig)
    plt.close(fig)
    
    return plots


def create_time_series_analysis(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    timestamps: Optional[list] = None,
) -> Dict[str, wandb.Image]:
    """
    Create time series plots showing prediction quality over time.
    Useful for seeing if model performs better/worse at certain times.
    """
    plots = {}
    
    # This would require temporal data
    # Placeholder for future implementation
    
    return plots


def create_skill_score_comparison(
    current_rmse: Dict[str, float],
    baseline_rmse: Dict[str, float],
) -> wandb.Image:
    """
    Compare model skill against baselines (climatology, persistence).
    
    Skill Score = 1 - (RMSE_model / RMSE_baseline)
    Positive = better than baseline
    """
    variables = list(current_rmse.keys())
    skill_scores = {}
    
    for var in variables:
        if var in baseline_rmse:
            skill = 1.0 - (current_rmse[var] / baseline_rmse[var])
            skill_scores[var] = skill
    
    if not skill_scores:
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    vars_list = list(skill_scores.keys())
    scores = list(skill_scores.values())
    colors = ['green' if s > 0 else 'red' for s in scores]
    
    bars = ax.barh(vars_list, scores, color=colors, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=2)
    ax.axvline(0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='50% improvement')
    ax.set_xlabel('Skill Score', fontweight='bold')
    ax.set_title('Model Skill vs Baseline (Climatology)', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        label = f'{score:.3f}'
        ax.text(score, i, label, ha='left' if score > 0 else 'right', 
               va='center', fontweight='bold')
    
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    
    return img


def _skewness(data: np.ndarray) -> float:
    """Compute skewness of data"""
    data = data.flatten()
    mean = data.mean()
    std = data.std()
    if std == 0:
        return 0.0
    return ((data - mean) ** 3).mean() / (std ** 3)


def create_variable_comparison_dashboard(
    metrics: Dict[str, Dict[str, float]],
    epoch: int,
) -> wandb.Image:
    """
    Create a comprehensive dashboard comparing all variables.
    
    Args:
        metrics: Dict of {variable_name: {metric_name: value}}
        epoch: Current epoch
    
    Returns:
        WandB image of dashboard
    """
    variables = list(metrics.keys())
    metric_names = list(metrics[variables[0]].keys()) if variables else []
    
    if not variables or not metric_names:
        return None
    
    # Create subplot grid
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metric_names):
        values = [metrics[var][metric] for var in variables]
        
        # Color code by performance
        if metric == 'rmse':
            colors = ['green' if v < 200 else 'orange' if v < 300 else 'red' for v in values]
        elif metric == 'correlation':
            colors = ['green' if v > 0.95 else 'orange' if v > 0.90 else 'red' for v in values]
        else:
            colors = 'skyblue'
        
        bars = ax.bar(variables, values, color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel(metric.upper(), fontweight='bold')
        ax.set_title(f'{metric.title()} by Variable', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2e}' if abs(value) < 0.01 else f'{value:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Rotate x labels
        ax.set_xticklabels(variables, rotation=45, ha='right')
    
    plt.suptitle(f'Multi-Variable Performance Dashboard (Epoch {epoch})', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    img = wandb.Image(fig)
    plt.close(fig)
    
    return img


def create_prediction_evolution_plot(
    rmse_history: Dict[str, list],
    corr_history: Dict[str, list],
) -> wandb.Image:
    """
    Plot how predictions evolve over epochs for each variable.
    
    Args:
        rmse_history: Dict of {variable: [rmse_epoch_0, rmse_epoch_1, ...]}
        corr_history: Dict of {variable: [corr_epoch_0, corr_epoch_1, ...]}
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot RMSE evolution
    for var, rmse_list in rmse_history.items():
        epochs = list(range(len(rmse_list)))
        ax1.plot(epochs, rmse_list, marker='o', linewidth=2, label=var, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('RMSE', fontweight='bold')
    ax1.set_title('RMSE Evolution During Training', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot Correlation evolution
    for var, corr_list in corr_history.items():
        epochs = list(range(len(corr_list)))
        ax2.plot(epochs, corr_list, marker='o', linewidth=2, label=var, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Correlation', fontweight='bold')
    ax2.set_title('Correlation Evolution During Training', fontweight='bold', fontsize=14)
    ax2.axhline(0.95, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target (0.95)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    img = wandb.Image(fig)
    plt.close(fig)
    
    return img


def log_physical_consistency_checks(
    predictions: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Check physical consistency of predictions.
    E.g., temperature in reasonable range, pressure positive, etc.
    """
    logs = {}
    
    # Temperature checks (should be in Kelvin, ~200-320K range)
    if '2m_temperature' in predictions:
        temp = predictions['2m_temperature']
        logs['physics/temp_min_K'] = temp.min().item()
        logs['physics/temp_max_K'] = temp.max().item()
        logs['physics/temp_mean_K'] = temp.mean().item()
        
        # Check for unphysical values
        unphysical_low = (temp < 180).sum().item()  # Below -93°C
        unphysical_high = (temp > 340).sum().item()  # Above 67°C
        logs['physics/temp_unphysical_count'] = unphysical_low + unphysical_high
        
        # Convert to Celsius for intuition
        temp_c = temp - 273.15
        logs['physics/temp_mean_C'] = temp_c.mean().item()
    
    # Pressure checks (should be ~50000-110000 Pa)
    if 'surface_pressure' in predictions:
        pres = predictions['surface_pressure']
        logs['physics/pressure_min_Pa'] = pres.min().item()
        logs['physics/pressure_max_Pa'] = pres.max().item()
        logs['physics/pressure_mean_Pa'] = pres.mean().item()
        
        # Check for unphysical values
        unphysical_low = (pres < 40000).sum().item()  # Below 400 hPa
        unphysical_high = (pres > 110000).sum().item()  # Above 1100 hPa
        logs['physics/pressure_unphysical_count'] = unphysical_low + unphysical_high
    
    return logs


def create_latitudinal_mean_plot(
    pred: torch.Tensor,
    target: torch.Tensor,
    var_name: str,
    epoch: int,
) -> wandb.Image:
    """
    Plot latitudinal mean (zonal mean) of predictions vs targets.
    Helps identify if model captures latitudinal gradients correctly.
    """
    # Compute zonal means (average over longitude)
    pred_zonal = pred.mean(dim=-1).mean(dim=0).cpu().numpy()  # [lat]
    target_zonal = target.mean(dim=-1).mean(dim=0).cpu().numpy()  # [lat]
    
    # Create latitude array (approximate)
    n_lat = len(pred_zonal)
    latitudes = np.linspace(-90, 90, n_lat)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(latitudes, target_zonal, linewidth=3, label='Target', color='black', alpha=0.7)
    ax.plot(latitudes, pred_zonal, linewidth=2, label='Predicted', color='red', linestyle='--')
    
    ax.set_xlabel('Latitude (degrees)', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'{var_name}', fontweight='bold', fontsize=12)
    ax.set_title(f'{var_name}: Zonal Mean (Epoch {epoch})', 
                fontweight='bold', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add reference lines
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)  # Equator
    ax.axvline(23.5, color='gray', linestyle=':', alpha=0.3)  # Tropics
    ax.axvline(-23.5, color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    
    img = wandb.Image(fig)
    plt.close(fig)
    
    return img





