"""
Comprehensive WandB Logging for Diffusion Models (EDM)

Specialized logging for diffusion models including:
- Sigma schedule visualization
- Denoising step progression
- Loss per noise level
- Sample diversity and quality metrics
- Reconstruction quality at different timesteps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer


class DiffusionWandBLogger(Callback):
    """
    Comprehensive WandB logging specifically for EDM and other diffusion models.
    
    Logs:
    1. Sigma schedule (noise level vs timestep)
    2. Denoising step visualizations (t=100, 50, 10, 1)
    3. Loss per noise level (sigma)
    4. Sample diversity metrics
    5. Reconstruction quality at different timesteps
    6. Energy plots and latent variance
    """
    
    def __init__(
        self,
        log_sigma_schedule: bool = True,
        log_denoising_steps: bool = True,
        log_loss_per_sigma: bool = True,
        log_sample_diversity: bool = True,
        log_every_n_steps: int = 100,
        log_every_n_epochs: int = 1,
        num_denoising_steps_to_log: List[int] = [100, 50, 10, 1],
    ):
        super().__init__()
        self.log_sigma_schedule = log_sigma_schedule
        self.log_denoising_steps = log_denoising_steps
        self.log_loss_per_sigma = log_loss_per_sigma
        self.log_sample_diversity = log_sample_diversity
        self.log_every_n_steps = log_every_n_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.num_denoising_steps_to_log = num_denoising_steps_to_log
        
        # Store denoising intermediates
        self.denoising_intermediates = {}
    
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """Log training metrics every N steps"""
        if batch_idx % self.log_every_n_steps != 0:
            return
        
        logs = {}
        
        # Log sigma schedule if available
        if self.log_sigma_schedule and hasattr(pl_module, 'diffusion_model'):
            logs.update(self._log_sigma_schedule(pl_module))
        
        # Log loss per sigma if available
        if self.log_loss_per_sigma and hasattr(pl_module, 'diffusion_model'):
            logs.update(self._log_loss_per_sigma(pl_module, outputs))
        
        # Log to wandb
        if logs and trainer.logger:
            trainer.logger.experiment.log(logs, step=trainer.global_step)
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Log comprehensive validation analysis for diffusion models"""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        if not trainer.logger:
            return
        
        logs = {}
        
        # 1. Sigma schedule visualization
        if self.log_sigma_schedule:
            logs.update(self._create_sigma_schedule_plot(pl_module))
        
        # 2. Denoising step visualizations
        if self.log_denoising_steps:
            logs.update(self._create_denoising_visualizations(pl_module))
        
        # 3. Sample diversity metrics
        if self.log_sample_diversity:
            logs.update(self._log_sample_diversity_metrics(pl_module))
        
        # Log to wandb
        if logs:
            trainer.logger.experiment.log(logs, step=trainer.global_step)
    
    def _log_sigma_schedule(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Log current sigma schedule"""
        logs = {}
        
        if hasattr(pl_module, 'diffusion_model'):
            diffusion = pl_module.diffusion_model
            
            # Get sigma parameters
            if hasattr(diffusion, 'sigma_min') and hasattr(diffusion, 'sigma_max_inf'):
                logs['diffusion/sigma_min'] = diffusion.sigma_min
                logs['diffusion/sigma_max'] = diffusion.sigma_max_inf
                
                # Generate sigma schedule
                if hasattr(diffusion, 'edm_discretization'):
                    steps = torch.linspace(0, 1, diffusion.hparams.num_steps)
                    sigmas = diffusion.edm_discretization(
                        steps=steps,
                        sigma_min=diffusion.sigma_min,
                        sigma_max=diffusion.sigma_max_inf
                    )
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(steps.cpu().numpy(), sigmas.cpu().numpy(), 
                           marker='o', linewidth=2, markersize=4)
                    ax.set_xlabel('Normalized Timestep (t)', fontweight='bold', fontsize=12)
                    ax.set_ylabel('Noise Level (σ)', fontweight='bold', fontsize=12)
                    ax.set_title('EDM Sigma Schedule', fontweight='bold', fontsize=14)
                    ax.set_yscale('log')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim([0, 1])
                    
                    # Add annotations
                    ax.axhline(diffusion.sigma_min, color='green', linestyle='--', 
                              alpha=0.5, label=f'σ_min={diffusion.sigma_min:.3f}')
                    ax.axhline(diffusion.sigma_max_inf, color='red', linestyle='--', 
                              alpha=0.5, label=f'σ_max={diffusion.sigma_max_inf:.1f}')
                    ax.legend()
                    
                    plt.tight_layout()
                    logs['diffusion/sigma_schedule'] = wandb.Image(fig)
                    plt.close(fig)
                    
                    # Log sigma values as table
                    sigma_table = wandb.Table(
                        columns=['timestep', 'sigma'],
                        data=[[float(t), float(s)] for t, s in zip(steps, sigmas)]
                    )
                    logs['diffusion/sigma_table'] = sigma_table
        
        return logs
    
    def _create_sigma_schedule_plot(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Create comprehensive sigma schedule visualization"""
        logs = {}
        
        if hasattr(pl_module, 'diffusion_model'):
            diffusion = pl_module.diffusion_model
            
            if hasattr(diffusion, 'edm_discretization'):
                # Generate full schedule
                num_points = 200
                steps = torch.linspace(0, 1, num_points)
                sigmas = diffusion.edm_discretization(
                    steps=steps,
                    sigma_min=diffusion.sigma_min,
                    sigma_max=diffusion.sigma_max_inf
                )
                
                # Create comprehensive plot
                fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                
                # Plot 1: Linear scale
                axes[0].plot(steps.cpu().numpy(), sigmas.cpu().numpy(), 
                           linewidth=2, color='blue')
                axes[0].set_xlabel('Normalized Timestep', fontweight='bold')
                axes[0].set_ylabel('Noise Level (σ)', fontweight='bold')
                axes[0].set_title('Sigma Schedule (Linear Scale)', fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: Log scale
                axes[1].plot(steps.cpu().numpy(), sigmas.cpu().numpy(), 
                           linewidth=2, color='red')
                axes[1].set_xlabel('Normalized Timestep', fontweight='bold')
                axes[1].set_ylabel('Noise Level (σ)', fontweight='bold')
                axes[1].set_title('Sigma Schedule (Log Scale)', fontweight='bold')
                axes[1].set_yscale('log')
                axes[1].grid(True, alpha=0.3)
                
                # Add parameter info
                info_text = f"σ_min={diffusion.sigma_min:.3f}, σ_max={diffusion.sigma_max_inf:.1f}, "
                if hasattr(diffusion, 'hparams'):
                    if hasattr(diffusion.hparams, 'rho'):
                        info_text += f"ρ={diffusion.hparams.rho}"
                    if hasattr(diffusion.hparams, 'P_mean'):
                        info_text += f", P_mean={diffusion.hparams.P_mean}"
                
                fig.suptitle(f'EDM Noise Schedule\n{info_text}', 
                           fontsize=16, fontweight='bold', y=0.995)
                plt.tight_layout()
                
                logs['diffusion/sigma_schedule_comprehensive'] = wandb.Image(fig)
                plt.close(fig)
        
        return logs
    
    def _log_loss_per_sigma(self, pl_module: LightningModule, outputs: Any) -> Dict[str, Any]:
        """Log loss breakdown by noise level"""
        logs = {}
        
        # Check if outputs contain loss per sigma
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if 'per_noise_level' in key or 'per_sigma' in key:
                    if torch.is_tensor(value):
                        logs[f'diffusion/{key}'] = value.item()
                    else:
                        logs[f'diffusion/{key}'] = value
        
        return logs
    
    def _create_denoising_visualizations(self, pl_module: LightningModule) -> Dict[str, Any]:
        """
        Create denoising step visualizations.
        This would require hooking into the sampling process.
        Placeholder for now - would need to modify sampling code.
        """
        logs = {}
        
        # This would be implemented by modifying the sample() method
        # to yield intermediate denoising steps
        # For now, log a placeholder message
        
        return logs
    
    def _log_sample_diversity_metrics(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Log metrics about sample diversity and quality"""
        logs = {}
        
        # This would compute diversity metrics if multiple samples are generated
        # Placeholder for now
        
        return logs


def create_denoising_progression_plot(
    denoising_steps: Dict[int, torch.Tensor],
    var_name: str,
    epoch: int,
) -> wandb.Image:
    """
    Create a visualization showing denoising progression.
    
    Args:
        denoising_steps: Dict of {timestep: tensor} showing denoised samples
        var_name: Variable name (e.g., "2m_temperature")
        epoch: Current epoch
    """
    if not denoising_steps:
        return None
    
    # Sort timesteps in descending order (high noise to low noise)
    timesteps = sorted(denoising_steps.keys(), reverse=True)
    
    n_steps = len(timesteps)
    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
    if n_steps == 1:
        axes = [axes]
    
    for ax, t in zip(axes, timesteps):
        sample = denoising_steps[t][0].cpu().numpy()  # Take first sample
        
        im = ax.imshow(sample, cmap='viridis')
        ax.set_title(f't={t}', fontweight='bold', fontsize=12)
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle(f'{var_name}: Denoising Progression (Epoch {epoch})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    img = wandb.Image(fig)
    plt.close(fig)
    
    return img


def create_loss_vs_sigma_plot(
    loss_per_sigma: Dict[float, float],
) -> wandb.Image:
    """
    Create plot showing loss as function of noise level (sigma).
    
    Args:
        loss_per_sigma: Dict of {sigma: loss_value}
    """
    if not loss_per_sigma:
        return None
    
    sigmas = sorted(loss_per_sigma.keys())
    losses = [loss_per_sigma[s] for s in sigmas]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sigmas, losses, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Noise Level (σ)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Loss', fontweight='bold', fontsize=12)
    ax.set_title('Loss vs Noise Level', fontweight='bold', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    img = wandb.Image(fig)
    plt.close(fig)
    
    return img


def create_energy_plot(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    var_name: str,
) -> wandb.Image:
    """
    Create energy spectrum plot comparing predictions and targets.
    
    Args:
        predictions: Model predictions [B, C, H, W]
        targets: Ground truth [B, C, H, W]
        var_name: Variable name
    """
    # Compute 2D FFT for energy spectrum
    pred_fft = torch.fft.fft2(predictions[0, 0].cpu())
    target_fft = torch.fft.fft2(targets[0, 0].cpu())
    
    pred_energy = torch.abs(pred_fft) ** 2
    target_energy = torch.abs(target_fft) ** 2
    
    # Compute radial average (energy vs wavenumber)
    h, w = pred_energy.shape
    center_y, center_x = h // 2, w // 2
    
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij'
    )
    r = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Bin by radius
    r_flat = r.flatten()
    pred_flat = pred_energy.flatten()
    target_flat = target_energy.flatten()
    
    max_r = int(r_flat.max())
    radii = np.arange(0, max_r, 1)
    pred_energy_radial = []
    target_energy_radial = []
    
    for rad in radii:
        mask = (r_flat >= rad) & (r_flat < rad + 1)
        pred_energy_radial.append(pred_flat[mask].mean().item())
        target_energy_radial.append(target_flat[mask].mean().item())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(radii, target_energy_radial, label='Target', linewidth=2, alpha=0.8)
    ax.plot(radii, pred_energy_radial, label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_xlabel('Wavenumber (k)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Energy (Power Spectral Density)', fontweight='bold', fontsize=12)
    ax.set_title(f'{var_name}: Energy Spectrum', fontweight='bold', fontsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    img = wandb.Image(fig)
    plt.close(fig)
    
    return img





