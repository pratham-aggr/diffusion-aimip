from __future__ import annotations

import time
from typing import Any, Dict, List, Callable, Optional

import torch
from tensordict import TensorDictBase
from torch import Tensor

from src.evaluation.aggregators._abstract_aggregator import _Aggregator
from src.experiment_types._base_experiment import BaseExperiment
from src.utilities.utils import (
    rrearrange, 
    torch_to_numpy, 
    clamp_raw_threshold_as_zero_in_normed
)
from src.utilities.climatebench_datamodule_utils import get_statistics

def predictions_post_process(output_vars: Any[List[str], str], threshold: Optional[float], normalization_statistics: Dict[str, float]) -> Callable[[Tensor], Tensor]:
    """
    Returns a post-processing function for model predictions.

    The returned function clamps the input predictions to a minimum threshold
    (and potentially other conditions defined in `clamp_tensor`).

    Args:
        min_threshold (Optional[float]): Minimum threshold to clamp the predictions.

    Returns:
        Callable[[Tensor], Tensor]: A function that takes predictions and returns the processed predictions.
    """
    pr_idx = output_vars.index("pr") if "pr" in output_vars else None
    
    def _post_processing(preds: Tensor) -> Tensor:
        if pr_idx is None or threshold is None:
            # If no precipitation index or threshold is provided, return predictions unchanged
            return preds
        # Clamp predictions using the specified minimum threshold
        preds[:, pr_idx, :, :] = clamp_raw_threshold_as_zero_in_normed(
            preds[:, pr_idx, :, :], threshold=threshold, mean=normalization_statistics["pr_mean"], std=normalization_statistics["pr_std"]
        )
        return preds

    return _post_processing


class EmulationExperiment(BaseExperiment):
    def __init__(
        self,
        pr_clamping: bool = False,
        pr_clamping_threshold: Optional[float] = None,
        return_outputs_at_evaluation: str | bool = False,  # can be "all", "preds_only", True, False
        **kwargs,
    ):
        super().__init__(**kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.return_outputs_at_evaluation
        self.save_hyperparameters(ignore=["model"])
        
        if pr_clamping:
            # Only get statistics if we need them for precipitation clamping
            data_mean_act, data_std_act = get_statistics(
                self.hparams.datamodule_config.output_vars, 
                self.hparams.datamodule_config.normalization_type,
                self.hparams.datamodule_config.precip_transform,
            )
            normalization_statistics = {
                "pr_mean": data_mean_act["pr"],
                "pr_std": data_std_act["pr"],
            }
            self.predictions_post_process = predictions_post_process(
                output_vars=self.hparams.datamodule_config.output_vars,
                threshold=pr_clamping_threshold, 
                normalization_statistics=normalization_statistics
            )
        else:
            self.predictions_post_process = None

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        # if we use the inputs as conditioning, and use an output-shaped input (e.g. for DDPM),
        # we need to use the output channels here!
        is_standard_diffusion = self.is_diffusion_model
        if is_standard_diffusion:
            return self.actual_num_output_channels(self.dims["output"])
        if self.stack_window_to_channel_dim:
            return num_input_channels * self.window
        return num_input_channels

    @property
    def num_conditional_channels(self) -> int:
        """For emulation diffusion models, conditioning includes both inputs and static fields."""
        nc = super().num_conditional_channels  # Get static condition channels
        if self.is_diffusion_model:
            # For emulation, inputs are passed as condition to the model during diffusion
            # So we need to account for both input channels and static condition channels
            nc += self.dims["input"]
        return nc

    @property
    def target_key(self) -> str:
        return "targets"

    @property
    def main_data_keys(self) -> List[str]:
        return ["inputs", self.target_key]

    @property
    def main_data_keys_train(self) -> List[str]:
        # Data is already normalized and packed from the datamodule for training
        return []  # Don't normalize during training
    
    @property
    def main_data_keys_val(self) -> List[str]:
        return []  # Don't normalize during validation

    @property
    def normalize_data_keys_val(self) -> List[str]:
        # Emulation data is already normalized and packed from the datamodule
        # Don't normalize again during evaluation
        return []
    
    @property
    def normalize_data_keys_train(self) -> List[str]:
        # Emulation data is already normalized and packed from the datamodule
        # Don't normalize again during training
        return []

    def training_step(self, batch: Any, batch_idx: int):
        """
        Override training_step to skip the normalization loop since data is already normalized from the datamodule.
        This is necessary because the base class iterates over self.main_data_keys and tries to normalize them.
        """
        # Skip the normalization loop (lines 1024-1031 in base class)
        # Data is already normalized and packed from the datamodule
        
        # Just call get_loss directly
        loss_output = self.get_loss(batch)
        
        # Handle dict or scalar loss output (copied from base class lines 1043-1052)
        if isinstance(loss_output, dict):
            # Flatten nested dicts (e.g. from compute_loss_per_sigma)
            # But skip non-scalar values like lists
            flat_dict = {}
            for k, v in loss_output.items():
                if isinstance(v, dict):
                    # Nested dict - flatten it
                    for k2, v2 in v.items():
                        if torch.is_tensor(v2):
                            if v2.numel() == 1:
                                flat_dict[f"{k}/{k2}"] = float(v2.item())
                        elif isinstance(v2, (int, float)):
                            flat_dict[f"{k}/{k2}"] = float(v2)
                        # Skip lists and other non-scalar types
                elif torch.is_tensor(v):
                    if v.numel() == 1:
                        flat_dict[k] = float(v.item())
                elif isinstance(v, (int, float)):
                    flat_dict[k] = float(v)
            
            if flat_dict:
                self.log_dict(flat_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            
            loss = loss_output.get("loss", loss_output)
            if isinstance(loss, dict):
                loss = loss["loss"]  # Handle nested case
            self.log("train/loss", float(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            loss = loss_output
            self.log("train/loss", float(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    @torch.inference_mode()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        return_outputs: bool | str = None,
        aggregators: Dict[str, _Aggregator] = None,
    ):
        start_time = time.time()
        return_dict = dict()
        return_outputs = return_outputs or self.hparams.return_outputs_at_evaluation

        # pop metadata from batch since not needed for evaluation
        metadata = batch.pop("metadata", None)
        
        # Compute raw_targets by denormalizing the normalized targets
        # Use .get() instead of .pop() to keep targets in batch for loss computation
        targets_normed = batch.get("targets", None)
        if targets_normed is not None:
            # Denormalize to get raw targets
            raw_targets = self.denormalize_batch(targets_normed, batch_key=self.target_key)
        else:
            raw_targets = None
        
        if self.is_diffusion_model:
            # log validation loss (targets still in batch)
            loss = self.get_loss(batch)
            aggregators["diffusion_loss"].update(loss=loss)
        
        # Now pop targets after loss computation
        batch.pop("targets", None)

        # Get predictions
        targets = self.get_target_variants(raw_targets, is_normalized=False)
        inputs_raw = batch.pop("inputs")
        inputs = self.transform_inputs(inputs_raw, split=split, ensemble=True)
        results, time_pred = self.time_it(self.predict, inputs, **batch)  # self.predict(inputs, **batch)

        # Return outputs and log metrics
        targets_raw, targets_normed = targets.pop("targets"), targets.pop("targets_normed")
        preds_raw, preds_normed = results.pop("preds"), results.pop("preds_normed")
        
        # Remove time dimension from targets if present (for single-step emulation)
        if torch.is_tensor(targets_normed) and len(targets_normed.shape) == 5:
            targets_normed = targets_normed[:, -1, ...]  # Take last timestep: [B, T, C, H, W] -> [B, C, H, W]
        if torch.is_tensor(targets_raw) and len(targets_raw.shape) == 5:
            targets_raw = targets_raw[:, -1, ...]
        
        # preds are returned as TensorDict from the model - need to convert to tensor
        # Handle ensemble predictions: if num_predictions > 1, predictions may have shape (N, B, ...)
        has_ensemble_dim = (torch.is_tensor(preds_raw) and len(preds_raw.shape) == 5 and 
                           preds_raw.shape[0] == self.num_predictions) or \
                          (isinstance(preds_raw, (dict, TensorDictBase)) and 
                           len(list(preds_raw.values())[0].shape) == 4 and
                           list(preds_raw.values())[0].shape[0] == self.num_predictions)
        
        if isinstance(preds_raw, (dict, TensorDictBase)):
            # Manually stack all variables to create packed tensor
            pred_list = [preds_raw[k] for k in sorted(preds_raw.keys())]
            first_pred_shape = pred_list[0].shape
            
            if has_ensemble_dim and len(first_pred_shape) == 4:
                # Each variable is (N, B, H, W) where N=num_predictions
                # Stack along dim=2 to create channel dimension: (N, B, H, W) each -> (N, B, C, H, W)
                preds_raw = torch.stack(pred_list, dim=2)  # (N, B, H, W) each -> (N, B, C, H, W)
            elif len(first_pred_shape) == 4:
                # Each variable is (B, H, W) - no ensemble dimension
                # Stack along channel dim: (B, H, W) -> (B, C, H, W)
                preds_raw = torch.stack(pred_list, dim=1)  # Stack along channel dim
            elif len(first_pred_shape) == 5:
                # Each variable is (N, B, C, H, W) or (B, T, C, H, W) - check first dim
                if first_pred_shape[0] == self.num_predictions:
                    # (N, B, C, H, W) - stack along channel dim
                    preds_raw = torch.stack(pred_list, dim=2)  # Stack along channel dim (dim=2 after N, B)
                else:
                    # (B, T, C, H, W) - take last timestep and stack
                    pred_list = [p[:, -1, ...] if len(p.shape) == 5 else p for p in pred_list]
                    preds_raw = torch.stack(pred_list, dim=1)
            else:
                # Each variable is (B, C, H, W) - already packed
                preds_raw = torch.stack(pred_list, dim=1)  # Stack along channel dim
        elif torch.is_tensor(preds_raw):
            # Already a tensor - check if it has ensemble dimension
            if has_ensemble_dim:
                # (N, B, C, H, W) - ensemble dimension present, keep it
                pass
            elif len(preds_raw.shape) == 5:
                # Might be (B, T, C, H, W) - take last timestep
                preds_raw = preds_raw[:, -1, ...]  # (B, T, C, H, W) -> (B, C, H, W)
        
        if isinstance(preds_normed, (dict, TensorDictBase)):
            pred_list = [preds_normed[k] for k in sorted(preds_normed.keys())]
            first_pred_shape = pred_list[0].shape
            has_ensemble_dim_normed = (len(first_pred_shape) == 4 and first_pred_shape[0] == self.num_predictions) or \
                                      (len(first_pred_shape) == 5 and first_pred_shape[0] == self.num_predictions)
            
            if has_ensemble_dim_normed and len(first_pred_shape) == 4:
                # Each variable is (N, B, H, W)
                preds_normed = torch.stack(pred_list, dim=2)  # (N, B, H, W) each -> (N, B, C, H, W)
            elif len(first_pred_shape) == 5 and first_pred_shape[0] == self.num_predictions:
                # Each variable is (N, B, C, H, W)
                preds_normed = torch.stack(pred_list, dim=2)  # Stack along channel dim
            elif len(first_pred_shape) == 4:
                # Each variable is (B, H, W) or (B, C, H, W)
                preds_normed = torch.stack(pred_list, dim=1)  # Stack along channel dim
            else:
                preds_normed = torch.stack(pred_list, dim=1)
        if return_outputs in [True, "all", "preds_only"]:
            return_dict["preds_normed"] = torch_to_numpy(preds_normed)
        if return_outputs in [True, "all"]:
            return_dict["targets_normed"] = torch_to_numpy(targets_normed)

        if return_outputs == "all":
            return_dict["targets"] = torch_to_numpy(targets)
            return_dict["preds"] = torch_to_numpy(preds_raw)
            # add remaining outputs
            return_dict.update({k: torch_to_numpy(v) for k, v in results.items()})

        # Compute metrics
        start_time_agg = time.time()
        
        # Simple RMSE for progress bar
        # Handle ensemble predictions: if num_predictions > 1, predictions are expanded
        if self.num_predictions > 1 and torch.is_tensor(preds_raw) and torch.is_tensor(targets_raw):
            # Check if preds_raw has expanded batch dimension (N*B instead of B)
            if preds_raw.shape[0] == targets_raw.shape[0] * self.num_predictions:
                # preds_raw is (N*B, C, H, W), targets_raw is (B, C, H, W)
                # Reshape preds_raw to (N, B, C, H, W) and take mean over ensemble
                B = targets_raw.shape[0]
                N = self.num_predictions
                preds_reshaped = preds_raw.reshape(N, B, *preds_raw.shape[1:])  # (N*B, C, H, W) -> (N, B, C, H, W)
                preds_mean = preds_reshaped.mean(dim=0)  # (N, B, C, H, W) -> (B, C, H, W)
                rmse = torch.sqrt(((preds_mean - targets_raw) ** 2).mean())
            elif len(preds_raw.shape) == 5 and preds_raw.shape[0] == self.num_predictions:
                # preds_raw is (N, B, C, H, W), targets_raw is (B, C, H, W)
                # Compute RMSE over ensemble mean
                preds_mean = preds_raw.mean(dim=0)  # (N, B, C, H, W) -> (B, C, H, W)
                rmse = torch.sqrt(((preds_mean - targets_raw) ** 2).mean())
            else:
                # Fallback: compute RMSE directly (shouldn't happen with num_predictions > 1)
                rmse = torch.sqrt(((preds_raw - targets_raw) ** 2).mean())
        else:
            # No ensemble dimension, compute RMSE directly
            rmse = torch.sqrt(((preds_raw - targets_raw) ** 2).mean())
        self.log(f"{split}/rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)
        
        # Feed aggregators - they need unpacked dict format
        # Unpack packed tensors back to dicts for variable-wise metrics
        # IMPORTANT: For CRPS metrics, we need to pass full ensemble (N, B, ...), not averaged
        # For other metrics, we can pass averaged (B, ...)
        # The aggregator's is_ensemble flag expects (N, B, ...) when True
        if self.num_predictions > 1 and torch.is_tensor(preds_raw):
            # Check if preds_raw has expanded batch dimension
            if preds_raw.shape[0] == targets_raw.shape[0] * self.num_predictions:
                # Reshape to (N, B, C, H, W) for proper handling
                B = targets_raw.shape[0]
                N = self.num_predictions
                preds_raw_for_unpack = preds_raw.reshape(N, B, *preds_raw.shape[1:])  # (N*B, C, H, W) -> (N, B, C, H, W)
                # For aggregators that need ensemble (CRPS, SSR), pass full ensemble
                # Unpack while preserving ensemble dimension: (N, B, C, H, W) -> dict of (N, B, H, W)
                # We need to unpack along channel dim (dim=2) for each ensemble member
                if hasattr(self.datamodule, 'out_packer') and self.datamodule.out_packer:
                    # Unpack each ensemble member separately, then stack
                    unpacked_list = []
                    for n in range(N):
                        pred_n = preds_raw_for_unpack[n]  # (B, C, H, W)
                        unpacked_n = self.datamodule.out_packer.unpack_simple(pred_n)  # dict of (B, H, W)
                        unpacked_list.append(unpacked_n)
                    # Stack along ensemble dimension: list of dicts -> dict of (N, B, H, W)
                    preds_raw_dict = {k: torch.stack([unpacked_list[n][k] for n in range(N)], dim=0) 
                                     for k in unpacked_list[0].keys()}
                else:
                    # No packer, keep as tensor with ensemble dimension
                    preds_raw_dict = preds_raw_for_unpack
            elif len(preds_raw.shape) == 5 and preds_raw.shape[0] == self.num_predictions:
                # (N, B, C, H, W) - already has ensemble dimension
                B = preds_raw.shape[1]
                N = preds_raw.shape[0]
                assert N == self.num_predictions, f"Ensemble dimension mismatch: {N} != {self.num_predictions}"
                assert B == targets_raw.shape[0], f"Batch dimension mismatch: {B} != {targets_raw.shape[0]}"
                if hasattr(self.datamodule, 'out_packer') and self.datamodule.out_packer:
                    # Unpack each ensemble member
                    unpacked_list = []
                    for n in range(N):
                        pred_n = preds_raw[n]  # (B, C, H, W)
                        assert len(pred_n.shape) == 4 and pred_n.shape[0] == B, \
                            f"After selecting ensemble member {n}, got shape {pred_n.shape}, expected (B={B}, C, H, W)"
                        unpacked_n = self.datamodule.out_packer.unpack_simple(pred_n)  # dict of (B, H, W)
                        # Verify unpacked shape
                        for k, v in unpacked_n.items():
                            if torch.is_tensor(v):
                                assert len(v.shape) == 3 and v.shape[0] == B, \
                                    f"Unpacked variable {k} has shape {v.shape}, expected (B={B}, H, W)"
                        unpacked_list.append(unpacked_n)
                    # Stack along ensemble dimension: list of dicts -> dict of (N, B, H, W)
                    preds_raw_dict = {k: torch.stack([unpacked_list[n][k] for n in range(N)], dim=0) 
                                     for k in unpacked_list[0].keys()}
                    # Verify final shape
                    for k, v in preds_raw_dict.items():
                        assert len(v.shape) == 4 and v.shape[0] == N and v.shape[1] == B, \
                            f"Final prediction {k} has shape {v.shape}, expected (N={N}, B={B}, H, W)"
                else:
                    preds_raw_dict = preds_raw
            else:
                # Fallback case: preds_raw might be (B, C, H, W) or have unexpected shape
                # If it's 4D, unpack normally. If it's 5D but doesn't match expected patterns, 
                # try to handle it by assuming first dim might be ensemble or time
                if len(preds_raw.shape) == 4:
                    # (B, C, H, W) - normal case, unpack along channel dim
                    preds_raw_dict = self.datamodule.out_packer.unpack_simple(preds_raw) if hasattr(self.datamodule, 'out_packer') and self.datamodule.out_packer else preds_raw
                elif len(preds_raw.shape) == 5:
                    # Unexpected 5D shape - try to handle by unpacking each "batch" element
                    # This shouldn't happen, but handle it defensively
                    raise RuntimeError(
                        f"Unexpected preds_raw shape {preds_raw.shape} when num_predictions={self.num_predictions}. "
                        f"Expected either (N*B, C, H, W) with N*B={targets_raw.shape[0] * self.num_predictions} "
                        f"or (N, B, C, H, W) with N={self.num_predictions}, but got shape {preds_raw.shape}."
                    )
                else:
                    preds_raw_dict = self.datamodule.out_packer.unpack_simple(preds_raw) if hasattr(self.datamodule, 'out_packer') and self.datamodule.out_packer else preds_raw
        else:
            # num_predictions == 1, normal unpacking
            preds_raw_dict = self.datamodule.out_packer.unpack_simple(preds_raw) if hasattr(self.datamodule, 'out_packer') and self.datamodule.out_packer else preds_raw
        
        # Same for normed predictions - preserve ensemble dimension for aggregators
        if self.num_predictions > 1 and torch.is_tensor(preds_normed):
            if preds_normed.shape[0] == targets_normed.shape[0] * self.num_predictions:
                B = targets_normed.shape[0]
                N = self.num_predictions
                preds_normed_for_unpack = preds_normed.reshape(N, B, *preds_normed.shape[1:])
                if hasattr(self.datamodule, 'out_packer') and self.datamodule.out_packer:
                    unpacked_list = []
                    for n in range(N):
                        pred_n = preds_normed_for_unpack[n]  # (B, C, H, W)
                        unpacked_n = self.datamodule.out_packer.unpack_simple(pred_n)  # dict of (B, H, W)
                        unpacked_list.append(unpacked_n)
                    preds_normed_dict = {k: torch.stack([unpacked_list[n][k] for n in range(N)], dim=0) 
                                       for k in unpacked_list[0].keys()}
                else:
                    preds_normed_dict = preds_normed_for_unpack
            elif len(preds_normed.shape) == 5 and preds_normed.shape[0] == self.num_predictions:
                B = preds_normed.shape[1]
                N = preds_normed.shape[0]
                if hasattr(self.datamodule, 'out_packer') and self.datamodule.out_packer:
                    unpacked_list = []
                    for n in range(N):
                        pred_n = preds_normed[n]  # (B, C, H, W)
                        unpacked_n = self.datamodule.out_packer.unpack_simple(pred_n)  # dict of (B, H, W)
                        unpacked_list.append(unpacked_n)
                    preds_normed_dict = {k: torch.stack([unpacked_list[n][k] for n in range(N)], dim=0) 
                                       for k in unpacked_list[0].keys()}
                else:
                    preds_normed_dict = preds_normed
            else:
                preds_normed_dict = self.datamodule.out_packer.unpack_simple(preds_normed) if hasattr(self.datamodule, 'out_packer') and self.datamodule.out_packer else preds_normed
        else:
            preds_normed_dict = self.datamodule.out_packer.unpack_simple(preds_normed) if hasattr(self.datamodule, 'out_packer') and self.datamodule.out_packer else preds_normed
        
        targets_raw_dict = self.datamodule.out_packer.unpack_simple(targets_raw) if hasattr(self.datamodule, 'out_packer') and self.datamodule.out_packer else targets_raw
        targets_normed_dict = self.datamodule.out_packer.unpack_simple(targets_normed) if hasattr(self.datamodule, 'out_packer') and self.datamodule.out_packer else targets_normed
        
        # Verify shapes before passing to aggregators
        # When num_predictions > 1, aggregators expect (N, B, H, W) for predictions
        if self.num_predictions > 1 and isinstance(preds_raw_dict, dict):
            B = targets_raw.shape[0]
            N = self.num_predictions
            for k, v in preds_raw_dict.items():
                if torch.is_tensor(v):
                    if len(v.shape) == 3:  # (B, H, W) - missing ensemble dimension
                        # Add ensemble dimension by repeating (for non-ensemble metrics, aggregator will handle it)
                        # Actually, we should have (N, B, H, W) already. If we have (B, H, W), something went wrong.
                        raise RuntimeError(
                            f"Prediction {k} has shape {v.shape} but expected (N={N}, B={B}, H, W) "
                            f"when num_predictions > 1. This suggests unpacking didn't preserve ensemble dimension."
                        )
                    elif len(v.shape) == 4:
                        if v.shape[0] != N or v.shape[1] != B:
                            raise RuntimeError(
                                f"Prediction {k} has shape {v.shape} but expected (N={N}, B={B}, H, W). "
                                f"Got (N={v.shape[0]}, B={v.shape[1]}, ...)"
                            )
        
        for agg_name, agg in aggregators.items():
            if agg_name == "diffusion_loss":
                continue  # already logged above
            _, time_agg = self.time_it(
                agg.update,
                target_data=targets_raw_dict,
                gen_data=preds_raw_dict,
                target_data_norm=targets_normed_dict,
                gen_data_norm=preds_normed_dict,
                metadata=metadata,
            )
        duration_agg = time.time() - start_time_agg
        duration_total = time.time() - start_time
        self.log_text.debug(f"Durations: total={duration_total:.2f}s, aggs={duration_agg:.2f}s, pred={time_pred:.2f}s")
        return return_dict

    def transform_inputs(self, inputs: Tensor, ensemble: bool, **kwargs) -> Tensor:
        inputs = self.pack_data(inputs, input_or_output="input")
        if self.stack_window_to_channel_dim:  # and inputs.shape[1] == self.window:
            inputs = rrearrange(inputs, "b window c lat lon -> b (window c) lat lon")
        elif len(inputs.shape) == 5:  # Has time dimension but not stacking to channels
            # For emulation, take only the last time step (most recent)
            inputs = inputs[:, -1, :, :, :]  # [b, window, c, lat, lon] -> [b, c, lat, lon]
        inputs = self.get_ensemble_inputs(inputs, **kwargs) if ensemble else inputs
        return inputs

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        split = "train" if self.training else "val"
        # Both inputs and targets are normalized
        inputs = self.transform_inputs(batch["inputs"], split=split, ensemble=False)
        targets = batch["targets"]
        
        # For single-step emulation, remove time dimension if present
        # Targets come as [B, T, C, H, W] but we only need [B, C, H, W]
        if torch.is_tensor(targets) and len(targets.shape) == 5:
            # Take the last time step (most recent target)
            targets = targets[:, -1, ...]
        
        # Uncomment below if targets is a dict of variable to variable target data
        targets = self.pack_data(targets, input_or_output="output")

        # Remove metadata from batch as not needed for loss computation
        batch.pop("metadata", None)

        extra_kwargs = {k: v for k, v in batch.items() if k not in ["inputs", "targets"]}
        extra_kwargs["predictions_post_process"] = self.predictions_post_process

        loss = self.model.get_loss(inputs=inputs, targets=targets, **extra_kwargs)
        return loss
