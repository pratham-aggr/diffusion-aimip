"""
Enhanced WandB Logging for Climate Model Emulation

Comprehensive visualizations including:
- Training dynamics (loss, gradients, learning rate)
- Model internals (weights, activations, attention maps)
- Predictions (spatial patterns, errors, statistics)
- Per-variable metrics and learning curves
- Gradient flow and stability metrics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer


class ComprehensiveWandBLogger(Callback):
    """
    Comprehensive WandB logging callback for climate emulation models.
    
    Logs:
    1. Training dynamics (loss curves, LR, gradient norms)
    2. Model internals (weight distributions, activation statistics)
    3. Predictions (spatial maps, error patterns, scatter plots)
    4. Per-variable metrics (RMSE, MAE, correlation by variable)
    5. Attention maps and feature visualizations
    6. Gradient flow and stability metrics
    """
    
    def __init__(
        self,
        log_gradient_flow: bool = True,
        log_weights: bool = True,
        log_activations: bool = True,
        log_attention_maps: bool = True,
        log_prediction_analysis: bool = True,
        log_spatial_patterns: bool = True,
        log_every_n_steps: int = 100,
        log_every_n_epochs: int = 1,
        max_samples_to_log: int = 4,
    ):
        super().__init__()
        self.log_gradient_flow = log_gradient_flow
        self.log_weights = log_weights
        self.log_activations = log_activations
        self.log_attention_maps = log_attention_maps
        self.log_prediction_analysis = log_prediction_analysis
        self.log_spatial_patterns = log_spatial_patterns
        self.log_every_n_steps = log_every_n_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples_to_log = max_samples_to_log
        
        # Store intermediate activations
        self.activations = {}
        self.attention_maps = {}
        
    def on_train_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs, 
        batch, 
        batch_idx: int
    ):
        """Log training metrics every N steps"""
        if batch_idx % self.log_every_n_steps != 0:
            return
            
        logs = {}
        
        # 1. Training dynamics
        if self.log_gradient_flow:
            logs.update(self._log_gradient_flow(pl_module))
        
        # 2. Learning rate per parameter group
        logs.update(self._log_learning_rates(pl_module))
        
        # 3. Model statistics
        if self.log_weights:
            logs.update(self._log_weight_statistics(pl_module))
        
        # Log to wandb
        if logs and trainer.logger:
            trainer.logger.experiment.log(logs, step=trainer.global_step)
    
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Log validation visualizations"""
        # Only log for first few batches to avoid overwhelming W&B
        if batch_idx >= self.max_samples_to_log:
            return
        
        # This will be called after validation_step, so outputs contain predictions
        # We'll log comprehensive analysis in on_validation_epoch_end
        pass
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Log comprehensive validation analysis"""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        logs = {}
        
        # 1. Model weight distributions
        if self.log_weights:
            logs.update(self._log_weight_distributions(pl_module))
        
        # 2. Attention visualizations (if available)
        if self.log_attention_maps and hasattr(pl_module.model, 'get_attention_maps'):
            logs.update(self._log_attention_visualizations(pl_module))
        
        # 3. Learning curves summary
        logs.update(self._create_learning_curves(trainer))
        
        # Log to wandb
        if logs and trainer.logger:
            trainer.logger.experiment.log(logs, step=trainer.global_step)
    
    def _log_gradient_flow(self, pl_module: LightningModule) -> Dict[str, Any]:
        """
        Log gradient flow through the network.
        Helps identify vanishing/exploding gradients.
        """
        logs = {}
        
        # Collect gradients from all parameters
        ave_grads = []
        max_grads = []
        layers = []
        
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                # Skip if no gradient
                grad = param.grad.detach()
                
                # Compute statistics
                ave_grads.append(grad.abs().mean().item())
                max_grads.append(grad.abs().max().item())
                
                # Simplified layer name
                layer_name = name.split('.')[0] if '.' in name else name
                layers.append(layer_name[:20])  # Truncate long names
                
                # Log per-layer gradient norms
                logs[f"gradients/norm/{name}"] = grad.norm().item()
        
        if ave_grads:
            # Create gradient flow plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.6, label='Mean')
            ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.6, label='Max')
            ax.hlines(0, 0, len(ave_grads) + 1, linewidth=2, color="k")
            ax.set_xticks(range(0, len(ave_grads), max(1, len(ave_grads) // 20)))
            ax.set_xticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers) // 20))], 
                              rotation=45, ha='right')
            ax.set_xlabel("Layers")
            ax.set_ylabel("Gradient Magnitude")
            ax.set_title("Gradient Flow Through Network")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            logs["gradients/flow_plot"] = wandb.Image(fig)
            plt.close(fig)
            
            # Summary statistics
            logs["gradients/mean_magnitude"] = np.mean(ave_grads)
            logs["gradients/max_magnitude"] = np.max(max_grads)
            logs["gradients/min_magnitude"] = np.min(ave_grads)
        
        return logs
    
    def _log_learning_rates(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Log learning rate for each parameter group"""
        logs = {}
        
        if hasattr(pl_module, 'optimizers'):
            optimizer = pl_module.optimizers()
            if optimizer is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    logs[f"lr/group_{i}"] = param_group['lr']
        
        return logs
    
    def _log_weight_statistics(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Log weight statistics (mean, std, min, max)"""
        logs = {}
        
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                weight = param.detach()
                logs[f"weights/mean/{name}"] = weight.mean().item()
                logs[f"weights/std/{name}"] = weight.std().item()
                logs[f"weights/min/{name}"] = weight.min().item()
                logs[f"weights/max/{name}"] = weight.max().item()
                
                # Check for NaN or Inf
                if torch.isnan(weight).any():
                    logs[f"weights/has_nan/{name}"] = 1.0
                if torch.isinf(weight).any():
                    logs[f"weights/has_inf/{name}"] = 1.0
        
        return logs
    
    def _log_weight_distributions(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Create histograms of weight distributions"""
        logs = {}
        
        # Sample a few key layers to avoid overwhelming W&B
        key_layers = ['enc', 'dec', 'out_conv']
        
        for name, param in pl_module.named_parameters():
            # Only log key layers
            if not any(layer in name for layer in key_layers):
                continue
                
            if param.requires_grad:
                weight = param.detach().cpu().numpy().flatten()
                
                # Create histogram
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(weight, bins=50, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Weight Distribution: {name}')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                logs[f"weights/distribution/{name}"] = wandb.Image(fig)
                plt.close(fig)
        
        return logs
    
    def _log_attention_visualizations(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Visualize attention maps if available"""
        logs = {}
        
        # This would require hooks to capture attention maps
        # Implementation depends on model architecture
        # Placeholder for future implementation
        
        return logs
    
    def _create_learning_curves(self, trainer: Trainer) -> Dict[str, Any]:
        """Create comprehensive learning curves"""
        logs = {}
        
        # Get metrics from trainer's logged metrics
        if hasattr(trainer, 'logged_metrics'):
            metrics = trainer.logged_metrics
            
            # Create a summary plot of key metrics
            key_metrics = ['train/loss', 'val/avg/rmse', 'val/rmse']
            available_metrics = [m for m in key_metrics if m in metrics]
            
            if available_metrics:
                fig, axes = plt.subplots(1, len(available_metrics), figsize=(6 * len(available_metrics), 4))
                if len(available_metrics) == 1:
                    axes = [axes]
                
                for ax, metric in zip(axes, available_metrics):
                    value = metrics[metric]
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    
                    ax.text(0.5, 0.5, f"{metric}\n{value:.4f}", 
                           ha='center', va='center', fontsize=14)
                    ax.set_title(metric)
                    ax.axis('off')
                
                plt.tight_layout()
                logs["metrics/summary"] = wandb.Image(fig)
                plt.close(fig)
        
        return logs


class DetailedPredictionLogger(Callback):
    """
    Log detailed prediction analysis including:
    - Spatial error patterns
    - Per-variable statistics
    - Prediction vs target scatter plots
    - Error histograms
    - Regional performance
    """
    
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        n_samples_to_visualize: int = 2,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.n_samples_to_visualize = n_samples_to_visualize
        self.validation_outputs = []
    
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Collect validation outputs for analysis"""
        if batch_idx < self.n_samples_to_visualize:
            self.validation_outputs.append({
                'batch': batch,
                'outputs': outputs,
                'batch_idx': batch_idx,
            })
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Create comprehensive prediction analysis"""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            self.validation_outputs = []
            return
        
        if not self.validation_outputs or not trainer.logger:
            self.validation_outputs = []
            return
        
        logs = {}
        
        # Analyze collected validation outputs
        for output_data in self.validation_outputs[:self.n_samples_to_visualize]:
            batch_logs = self._analyze_predictions(
                output_data, 
                pl_module, 
                trainer.current_epoch
            )
            logs.update(batch_logs)
        
        # Log to wandb
        if logs:
            trainer.logger.experiment.log(logs, step=trainer.global_step)
        
        # Clear stored outputs
        self.validation_outputs = []
    
    def _analyze_predictions(
        self, 
        output_data: Dict, 
        pl_module: LightningModule,
        epoch: int,
    ) -> Dict[str, Any]:
        """Analyze a single batch of predictions"""
        logs = {}
        batch_idx = output_data['batch_idx']
        
        # This is a placeholder - actual implementation would depend on
        # the structure of your validation outputs
        # You would typically extract predictions and targets here
        
        return logs


class ActivationMonitor(Callback):
    """
    Monitor activation statistics through the network.
    Helps identify dead neurons, saturation, and activation distribution shifts.
    """
    
    def __init__(self, log_every_n_epochs: int = 5):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.handles = []
        self.activations = {}
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        """Register hooks to capture activations"""
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        for name, module in pl_module.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Only monitor a subset of layers to avoid memory issues
                if any(key in name for key in ['enc', 'dec', 'out_conv']):
                    handle = module.register_forward_hook(get_activation(name))
                    self.handles.append(handle)
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Log activation statistics"""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        if not self.activations or not trainer.logger:
            return
        
        logs = {}
        
        for name, activation in self.activations.items():
            # Compute statistics
            act = activation.cpu()
            
            logs[f"activations/mean/{name}"] = act.mean().item()
            logs[f"activations/std/{name}"] = act.std().item()
            logs[f"activations/max/{name}"] = act.max().item()
            logs[f"activations/min/{name}"] = act.min().item()
            
            # Percentage of dead neurons (activations == 0)
            dead_pct = (act == 0).float().mean().item() * 100
            logs[f"activations/dead_pct/{name}"] = dead_pct
            
            # Saturation (for ReLU/sigmoid)
            if act.max() > 0:
                saturated_pct = (act >= 0.99 * act.max()).float().mean().item() * 100
                logs[f"activations/saturated_pct/{name}"] = saturated_pct
        
        # Log to wandb
        if logs:
            trainer.logger.experiment.log(logs, step=trainer.global_step)
        
        # Clear activations to free memory
        self.activations = {}
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        """Remove hooks"""
        for handle in self.handles:
            handle.remove()

