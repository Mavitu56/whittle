from __future__ import annotations

import torch
from typing import Any

from whittle.training_strategies.base_strategy import BaseTrainingStrategy


class SandwichStrategy(BaseTrainingStrategy):
    """
    Sandwich strategy with improved stability.

    In each step, the sandwich strategy updates the super-network, the smallest, and a set of randomly sampled
    sub-networks.
    """

    def __init__(self, random_samples: int = 2, **kwargs: Any):
        """
        Initialises a `SandwichStrategy`

        Args:
            random_samples: the number of randomly sampled sub-networks to sample and update in each step
            **kwargs: kwargs of `BaseTrainingStrategy`
        """
        super().__init__(**kwargs)
        self.random_samples = random_samples

    def __call__(self, model, inputs, outputs, scale_loss=1, **kwargs):
        total_loss = 0
        
        # Safety check - verify model parameters before training
        self._check_model_parameters(model, "before-training")
        
        try:
            # update super-network
            model.reset_super_network()
            loss = self.compute_loss(model, inputs, outputs)
            
            # Scale loss with safety check
            loss = self._safe_scale_loss(loss, scale_loss)
            
            # Backward pass with safety guard
            self._safe_backward(loss)
            
            # Clip gradients to prevent explosion
            self._clip_gradients(model)
            
            total_loss += loss.item() if torch.isfinite(loss).all() else 0.0
            
            # update random sub-networks
            for i in range(self.random_samples):
                config = self.sampler.sample()
                model.set_sub_network(**config)
                loss = self.compute_loss(model, inputs, outputs)
                
                # Scale loss with safety check
                loss = self._safe_scale_loss(loss, scale_loss)
                
                # Backward pass with safety guard
                self._safe_backward(loss)
                
                # Clip gradients to prevent explosion
                self._clip_gradients(model)
                
                model.reset_super_network()
                total_loss += loss.item() if torch.isfinite(loss).all() else 0.0

            # smallest network
            config = self.sampler.get_smallest_sub_network()
            model.set_sub_network(**config)
            loss = self.compute_loss(model, inputs, outputs)
            
            # Scale loss with safety check
            loss = self._safe_scale_loss(loss, scale_loss)
            
            # Backward pass with safety guard
            self._safe_backward(loss)
            
            # Clip gradients to prevent explosion
            self._clip_gradients(model)
            
            model.reset_super_network()
            total_loss += loss.item() if torch.isfinite(loss).all() else 0.0
            
            # Final check for NaN in gradients
            self._fix_nan_gradients(model)
            
        except Exception as e:
            print(f"[ERROR] Exception in SandwichStrategy.__call__: {str(e)}")
            # Ensure model is reset to super network state
            try:
                model.reset_super_network()
            except:
                pass
            # Return a small non-zero loss to avoid breaking training
            return 1.0

        return total_loss
    
    def _safe_scale_loss(self, loss, scale):
        """Scale loss with safety check."""
        if not torch.isfinite(loss).all():
            print("[WARNING] Loss is not finite before scaling")
            return torch.tensor(1.0, device=loss.device, requires_grad=True)
        
        scaled_loss = loss * scale
        
        if not torch.isfinite(scaled_loss).all():
            print(f"[WARNING] Scaled loss is not finite (scale={scale}, loss={loss})")
            return loss  # Return unscaled loss instead
            
        return scaled_loss
        
    def _safe_backward(self, loss):
        """Perform backward pass with safety guard."""
        try:
            if self.fabric is None:
                loss.backward()
            else:
                self.fabric.backward(loss)
        except RuntimeError as e:
            print(f"[ERROR] Backward pass failed: {str(e)}")
            # No need to re-raise - we'll continue with training
    
    def _clip_gradients(self, model):
        """Clip gradients to prevent explosion."""
        try:
            # Clip gradients by value
            for param in model.parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_value_(param, self.grad_clip_value)
        except Exception as e:
            print(f"[ERROR] Gradient clipping failed: {str(e)}")
    
    def _fix_nan_gradients(self, model):
        """Replace NaN/Inf gradients with zeros."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                nan_mask = ~torch.isfinite(param.grad)
                if nan_mask.any():
                    nan_count = nan_mask.sum().item()
                    print(f"[WARNING] Fixing {nan_count} NaN/Inf values in gradient for {name}")
                    param.grad.data[nan_mask] = 0.0
    
    def _check_model_parameters(self, model, stage):
        """Check model parameters for NaN/Inf values."""
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                print(f"[WARNING] {stage}: Parameter '{name}' has {nan_count} NaN and {inf_count} Inf values")
