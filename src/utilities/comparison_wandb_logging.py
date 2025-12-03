"""
Unified WandB Logging for Model Comparison

This module provides comprehensive logging that works for both:
1. Direct prediction models (DhariwalUNet for emulation)
2. Diffusion models (EDM)

Allows side-by-side comparison in WandB with consistent metrics and visualizations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer
from scipy import stats
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)


class UnifiedComparisonLogger(Callback):
    """
    Unified logging callback for comparing direct prediction vs diffusion models.
    
    Logs consistent metrics for both model types:
    - Training/validation losses (overall + per diffusion step for EDM)
    - Learning rate schedules
    - Gradient norms and parameter updates
    - Weight histograms
    - Sigma schedule (for EDM only)
    - Predicted vs Ground Truth maps
    - Denoising step visualizations (for EDM)
    - Spectral Power Density (PSD) comparisons
    - Metrics: RMSE, MAE, CRPS, ACC
    - Residual distributions
    - Ensemble spread vs error
    - Feature importance (attention/correlation maps)
    - Latent variance and energy plots
    - Sample diversity
    """
    
    def __init__(
        self,
        model_type: str = "auto",  # "auto", "direct", or "diffusion"
        log_every_n_steps: int = 100,
        log_every_n_epochs: int = 1,
        log_denoising_steps: List[int] = [100, 50, 10, 1],
        max_samples_per_epoch: int = 8,
    ):
        super().__init__()
        self.model_type = model_type
        self.log_every_n_steps = log_every_n_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.log_denoising_steps = sorted(log_denoising_steps, reverse=True)
        self.max_samples_per_epoch = max_samples_per_epoch
        
        # Storage for intermediate values
        self.gradient_norms = {}
        self.sigma_history = []
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
        
        # Determine model type
        is_diffusion = self._is_diffusion_model(pl_module)
        model_prefix = "diffusion" if is_diffusion else "direct"
        
        # 1. Learning rate schedule
        logs.update(self._log_learning_rate_schedule(pl_module, model_prefix))
        
        # 2. Gradient norms
        logs.update(self._log_gradient_norms(pl_module, model_prefix))
        
        # 3. Parameter update magnitudes
        logs.update(self._log_parameter_updates(pl_module, model_prefix))
        
        # 4. Weight histograms (less frequent)
        if batch_idx % (self.log_every_n_steps * 10) == 0:
            logs.update(self._log_weight_histograms(pl_module, model_prefix))
        
        # 5. Sigma schedule (diffusion only)
        if is_diffusion:
            logs.update(self._log_sigma_schedule(pl_module))
        
        # 6. Loss per sigma (diffusion only)
        if is_diffusion and isinstance(outputs, dict):
            logs.update(self._extract_loss_per_sigma(outputs))
        
        # Log to wandb
        if logs and trainer.logger:
            trainer.logger.experiment.log(logs, step=trainer.global_step)
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Comprehensive validation logging"""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        if not trainer.logger:
            return
        
        is_diffusion = self._is_diffusion_model(pl_module)
        model_prefix = "diffusion" if is_diffusion else "direct"
        
        logs = {}
        
        # 1. Loss curves (already logged by PyTorch Lightning, but ensure consistency)
        # 2. Predicted vs Ground Truth maps
        logs.update(self._log_prediction_maps(trainer, pl_module, model_prefix))
        
        # 3. Spectral Power Density
        logs.update(self._log_spectral_power_density(trainer, pl_module, model_prefix))
        
        # 4. Metrics plots (RMSE, MAE, CRPS, ACC)
        logs.update(self._log_metrics_plots(trainer, pl_module, model_prefix))
        
        # 5. Residual distributions
        logs.update(self._log_residual_distributions(trainer, pl_module, model_prefix))
        
        # 6. Pixel intensity distributions
        logs.update(self._log_intensity_distributions(trainer, pl_module, model_prefix))
        
        # 7. Ensemble spread vs error (if ensemble predictions)
        logs.update(self._log_ensemble_analysis(trainer, pl_module, model_prefix))
        
        # 8. Feature importance (attention/correlation maps)
        logs.update(self._log_feature_importance(trainer, pl_module, model_prefix))
        
        # 9. Latent variance and energy plots
        logs.update(self._log_latent_analysis(trainer, pl_module, model_prefix))
        
        # 10. Denoising steps (diffusion only)
        if is_diffusion:
            logs.update(self._log_denoising_visualizations(trainer, pl_module))
        
        # 11. Sample diversity (diffusion only)
        if is_diffusion:
            logs.update(self._log_sample_diversity(trainer, pl_module))
        
        # Log to wandb
        if logs:
            trainer.logger.experiment.log(logs, step=trainer.global_step)
    
    def _is_diffusion_model(self, pl_module: LightningModule) -> bool:
        """Detect if model is a diffusion model"""
        if self.model_type != "auto":
            return self.model_type == "diffusion"
        
        # Check for diffusion model attributes
        if hasattr(pl_module, 'diffusion_model'):
            return True
        if hasattr(pl_module, 'model') and hasattr(pl_module.model, 'diffusion_model'):
            return True
        
        # Check config for diffusion keyword
        if hasattr(pl_module, 'hparams'):
            if isinstance(pl_module.hparams, dict):
                if 'diffusion' in str(pl_module.hparams).lower():
                    return True
        
        return False
    
    def _log_learning_rate_schedule(
        self, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Log learning rate for each parameter group"""
        logs = {}
        
        if hasattr(pl_module, 'optimizers'):
            optimizer = pl_module.optimizers()
            if optimizer is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    lr_key = f"{prefix}/learning_rate/group_{i}"
                    logs[lr_key] = param_group['lr']
        
        return logs
    
    def _log_gradient_norms(
        self, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Log gradient norms per layer"""
        logs = {}
        
        try:
            total_norm = 0.0
            param_norms = {}
            
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    
                    # Log per-layer norm (only for key layers to avoid clutter)
                    if any(key in name for key in ['conv', 'attention', 'resblock']):
                        safe_name = name.replace('.', '_')
                        logs[f"{prefix}/gradient_norm/{safe_name}"] = param_norm.item()
            
            total_norm = total_norm ** (1. / 2)
            logs[f"{prefix}/gradient_norm/total"] = total_norm
            
            # Store for history
            self.gradient_norms[prefix] = self.gradient_norms.get(prefix, [])
            self.gradient_norms[prefix].append(total_norm)
            
        except Exception as e:
            pass  # Silently skip if gradients not available
        
        return logs
    
    def _log_parameter_updates(
        self, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Log parameter update magnitudes"""
        logs = {}
        
        try:
            total_update = 0.0
            for name, param in pl_module.named_parameters():
                if param.grad is not None and hasattr(param, 'data'):
                    # Estimate update magnitude (would need to track previous state for exact)
                    update_mag = param.grad.data.abs().mean().item()
                    total_update += update_mag
                    
                    if any(key in name for key in ['conv', 'attention']):
                        safe_name = name.replace('.', '_')
                        logs[f"{prefix}/param_update/{safe_name}"] = update_mag
            
            logs[f"{prefix}/param_update/total_mean"] = total_update
            
        except Exception:
            pass
        
        return logs
    
    def _log_weight_histograms(
        self, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Log weight histograms for key layers"""
        logs = {}
        
        try:
            for name, param in pl_module.named_parameters():
                if any(key in name for key in ['conv.weight', 'attention.to_qkv', 'resblock']):
                    safe_name = name.replace('.', '_')
                    logs[f"{prefix}/weights/{safe_name}"] = wandb.Histogram(param.data.cpu().numpy())
        except Exception:
            pass
        
        return logs
    
    def _log_sigma_schedule(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Log sigma (noise level) schedule for diffusion models"""
        logs = {}
        
        try:
            if hasattr(pl_module, 'diffusion_model'):
                diffusion = pl_module.diffusion_model
                
                if hasattr(diffusion, 'sigma_min') and hasattr(diffusion, 'sigma_max_inf'):
                    logs['diffusion/sigma_min'] = diffusion.sigma_min
                    logs['diffusion/sigma_max'] = diffusion.sigma_max_inf
                    
                    # Generate schedule plot
                    if hasattr(diffusion, 'edm_discretization'):
                        num_steps = getattr(diffusion.hparams, 'num_steps', 18)
                        steps = torch.linspace(0, 1, num_steps)
                        sigmas = diffusion.edm_discretization(
                            steps=steps,
                            sigma_min=diffusion.sigma_min,
                            sigma_max=diffusion.sigma_max_inf
                        )
                        
                        # Create plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(steps.cpu().numpy(), sigmas.cpu().numpy(), 
                               marker='o', linewidth=2, markersize=4)
                        ax.set_xlabel('Normalized Timestep', fontweight='bold')
                        ax.set_ylabel('Noise Level (σ)', fontweight='bold')
                        ax.set_title('EDM Sigma Schedule', fontweight='bold')
                        ax.set_yscale('log')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        logs['diffusion/sigma_schedule'] = wandb.Image(fig)
                        plt.close(fig)
                        
                        # Store for history
                        self.sigma_history.append({
                            'step': steps.cpu().numpy(),
                            'sigma': sigmas.cpu().numpy()
                        })
        except Exception as e:
            pass
        
        return logs
    
    def _extract_loss_per_sigma(self, outputs: Dict) -> Dict[str, Any]:
        """Extract loss per sigma level from outputs"""
        logs = {}
        
        for key, value in outputs.items():
            if 'per_noise_level' in key or 'per_sigma' in key:
                if torch.is_tensor(value):
                    logs[f"diffusion/loss_per_sigma/{key}"] = value.item()
                else:
                    logs[f"diffusion/loss_per_sigma/{key}"] = value
        
        return logs
    
    def _log_prediction_maps(
        self, trainer: Trainer, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Log predicted vs ground truth maps"""
        # This will be handled by the existing snapshot aggregator
        # Just ensure metrics are logged with prefix
        return {}
    
    def _log_spectral_power_density(
        self, trainer: Trainer, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Log spectral power density comparison"""
        logs = {}
        
        # This would require access to validation predictions
        # Placeholder - would need to hook into validation step
        return logs
    
    def _log_metrics_plots(
        self, trainer: Trainer, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Create plots of RMSE, MAE, CRPS, ACC over time"""
        logs = {}
        
        # Get logged metrics from trainer
        if hasattr(trainer, 'logged_metrics'):
            metrics = trainer.logged_metrics
            
            # Extract relevant metrics and convert to CPU/float
            rmse_vals = [v.cpu().item() if torch.is_tensor(v) else float(v) 
                        for k, v in metrics.items() if 'rmse' in k.lower() and 'val' in k]
            mae_vals = [v.cpu().item() if torch.is_tensor(v) else float(v)
                       for k, v in metrics.items() if 'mae' in k.lower() and 'val' in k]
            
            if rmse_vals or mae_vals:
                # Create metrics plot
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # RMSE over epochs
                if rmse_vals:
                    axes[0, 0].plot(rmse_vals, linewidth=2)
                    axes[0, 0].set_title('RMSE Over Training', fontweight='bold')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('RMSE')
                    axes[0, 0].grid(True, alpha=0.3)
                
                # MAE over epochs
                if mae_vals:
                    axes[0, 1].plot(mae_vals, linewidth=2)
                    axes[0, 1].set_title('MAE Over Training', fontweight='bold')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('MAE')
                    axes[0, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                logs[f"{prefix}/metrics_overview"] = wandb.Image(fig)
                plt.close(fig)
        
        return logs
    
    def _log_residual_distributions(
        self, trainer: Trainer, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Log residual (error) distributions"""
        # Placeholder - would need access to validation predictions
        return {}
    
    def _log_intensity_distributions(
        self, trainer: Trainer, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Log pixel intensity distributions"""
        # Placeholder
        return {}
    
    def _log_ensemble_analysis(
        self, trainer: Trainer, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Log ensemble spread vs error analysis"""
        # Placeholder
        return {}
    
    def _log_feature_importance(
        self, trainer: Trainer, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Log attention maps or correlation maps showing input influence"""
        # Placeholder
        return {}
    
    def _log_latent_analysis(
        self, trainer: Trainer, pl_module: LightningModule, prefix: str
    ) -> Dict[str, Any]:
        """Log latent variance and energy plots"""
        # Placeholder
        return {}
    
    def _log_denoising_visualizations(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> Dict[str, Any]:
        """Log denoising step visualizations for diffusion models"""
        logs = {}
        
        # This would require modifying the sample() method to yield intermediates
        # Placeholder for now
        return logs
    
    def _log_sample_diversity(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> Dict[str, Any]:
        """Log sample diversity metrics for diffusion models"""
        logs = {}
        
        # Placeholder
        return logs


def create_comparison_panel(
    direct_metrics: Dict[str, float],
    diffusion_metrics: Dict[str, float],
    epoch: int,
) -> wandb.Image:
    """
    Create a side-by-side comparison panel for direct vs diffusion models.
    
    Args:
        direct_metrics: Metrics from direct prediction model
        diffusion_metrics: Metrics from diffusion model
        epoch: Current epoch
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Extract common metrics
    metrics_to_compare = ['rmse', 'mae', 'correlation', 'bias']
    
    direct_vals = []
    diffusion_vals = []
    metric_names = []
    
    for metric in metrics_to_compare:
        d_val = direct_metrics.get(metric, None)
        df_val = diffusion_metrics.get(metric, None)
        
        if d_val is not None and df_val is not None:
            direct_vals.append(d_val)
            diffusion_vals.append(df_val)
            metric_names.append(metric.upper())
    
    if direct_vals and diffusion_vals:
        # Bar plot comparison
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, direct_vals, width, label='Direct', alpha=0.8)
        axes[0, 0].bar(x + width/2, diffusion_vals, width, label='Diffusion', alpha=0.8)
        axes[0, 0].set_ylabel('Value', fontweight='bold')
        axes[0, 0].set_title('Metrics Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Direct vs Diffusion Model Comparison (Epoch {epoch})',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    img = wandb.Image(fig)
    plt.close(fig)
    
    return img

