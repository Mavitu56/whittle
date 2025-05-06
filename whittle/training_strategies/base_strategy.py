from __future__ import annotations

from collections.abc import Callable
import logging
import time
from pathlib import Path

import torch
from lightning.fabric import Fabric

from whittle.loss import DistillLoss
from whittle.sampling.random_sampler import BaseSampler

# Configure logging for NaN detection
logger = logging.getLogger("NaNDebug")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    # Also add file handler
    try:
        file_handler = logging.FileHandler("nan_debug.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    except Exception:
        # If file cannot be created, continue without file logging
        pass


class NaNDetector:
    """
    Utility class to detect and debug NaN/Inf values in model training.
    
    Features:
    - Checks inputs, outputs, gradients, and parameters for NaN/Inf
    - Logs detailed information when NaN/Inf is detected
    - Can automatically save state for debugging
    - Provides hooks for monitoring model during training
    """
    
    def __init__(self, model, save_dir="nan_debug", max_dumps=3):
        """
        Initialize the NaN detector.
        
        Args:
            model: The PyTorch model to monitor
            save_dir: Directory to save debug information when NaN is detected
            max_dumps: Maximum number of state dumps to save (to avoid filling disk)
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.dump_count = 0
        self.max_dumps = max_dumps
        self.hooks = []
        self.debug_active = True
        
        # Add hooks to all layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to detect NaN values."""
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') or hasattr(module, 'bias'):
                # Register forward hook
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._forward_hook(mod, inp, out, name)
                )
                self.hooks.append(hook)
                
                # Register backward hook for gradients
                hook = module.register_full_backward_hook(
                    lambda mod, grad_in, grad_out, name=name: self._backward_hook(mod, grad_in, grad_out, name)
                )
                self.hooks.append(hook)
    
    def _forward_hook(self, module, inputs, outputs, name):
        """Hook for forward pass to detect NaN/Inf in inputs and outputs."""
        if not self.debug_active:
            return
            
        # Check inputs
        if inputs is not None:
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor) and not torch.isfinite(inp).all():
                    logger.error(f"NaN/Inf detected in input {i} to module {name}")
                    self._dump_state(f"nan_input_{name}")
        
        # Check outputs
        if outputs is not None:
            if isinstance(outputs, torch.Tensor) and not torch.isfinite(outputs).all():
                logger.error(f"NaN/Inf detected in output of module {name}")
                self._dump_state(f"nan_output_{name}")
    
    def _backward_hook(self, module, grad_inputs, grad_outputs, name):
        """Hook for backward pass to detect NaN/Inf in gradients."""
        if not self.debug_active:
            return
            
        # Check gradient inputs
        if grad_inputs is not None:
            for i, grad in enumerate(grad_inputs):
                if isinstance(grad, torch.Tensor) and grad is not None and not torch.isfinite(grad).all():
                    logger.error(f"NaN/Inf detected in gradient input {i} of module {name}")
                    self._dump_state(f"nan_grad_input_{name}")
        
        # Check gradient outputs
        if grad_outputs is not None:
            for i, grad in enumerate(grad_outputs):
                if isinstance(grad, torch.Tensor) and grad is not None and not torch.isfinite(grad).all():
                    logger.error(f"NaN/Inf detected in gradient output {i} of module {name}")
                    self._dump_state(f"nan_grad_output_{name}")
    
    def _dump_state(self, prefix):
        """Dump model state for debugging when NaN is detected."""
        if self.dump_count >= self.max_dumps:
            return
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dump_file = self.save_dir / f"{prefix}_{timestamp}.pt"
        
        # Create a state dictionary with model parameters and their gradients
        state = {
            'parameters': {},
            'gradients': {},
            'timestamp': timestamp
        }
        
        for name, param in self.model.named_parameters():
            state['parameters'][name] = param.detach().cpu()
            if param.grad is not None:
                state['gradients'][name] = param.grad.detach().cpu()
        
        torch.save(state, dump_file)
        logger.warning(f"Dumped model state to {dump_file}")
        self.dump_count += 1
    
    def check_batch(self, batch, prefix="batch"):
        """Check an input batch for NaN/Inf values."""
        if not self.debug_active:
            return True
            
        has_nan = False
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                logger.error(f"NaN/Inf detected in batch['{key}']")
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                logger.error(f"  NaN count: {nan_count}, Inf count: {inf_count}")
                has_nan = True
                
                # Sample of positions where NaNs occur
                if nan_count > 0:
                    nan_indices = torch.where(torch.isnan(tensor))
                    sample_indices = [idx[:min(5, len(idx))] for idx in nan_indices]
                    logger.error(f"  NaN sample positions: {sample_indices}")
                
                # Save the problematic tensor
                if self.dump_count < self.max_dumps:
                    dump_file = self.save_dir / f"{prefix}_{key}_nan_{time.strftime('%Y%m%d-%H%M%S')}.pt"
                    torch.save(tensor.detach().cpu(), dump_file)
                    self.dump_count += 1
        
        return not has_nan
    
    def check_loss(self, loss):
        """Check if loss is NaN/Inf and log detailed information."""
        if not self.debug_active:
            return True
            
        if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
            logger.error(f"NaN/Inf detected in loss: {loss.item() if loss.numel() == 1 else loss}")
            self._dump_state("nan_loss")
            return False
        return True
    
    def check_model_parameters(self):
        """Check all model parameters for NaN/Inf values."""
        if not self.debug_active:
            return True
            
        has_nan = False
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                logger.error(f"NaN/Inf detected in parameter {name}")
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                logger.error(f"  NaN count: {nan_count}, Inf count: {inf_count}")
                has_nan = True
                
                if self.dump_count < self.max_dumps:
                    dump_file = self.save_dir / f"param_{name}_nan_{time.strftime('%Y%m%d-%H%M%S')}.pt"
                    torch.save(param.detach().cpu(), dump_file)
                    self.dump_count += 1
        
        return not has_nan
    
    def check_gradients(self):
        """Check all model gradients for NaN/Inf values."""
        if not self.debug_active:
            return True
            
        has_nan = False
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                logger.error(f"NaN/Inf detected in gradient for parameter {name}")
                nan_count = torch.isnan(param.grad).sum().item()
                inf_count = torch.isinf(param.grad).sum().item()
                logger.error(f"  NaN count: {nan_count}, Inf count: {inf_count}")
                
                # Log statistics of the parameter and its gradient
                if param.numel() > 0:
                    if torch.isfinite(param).all():
                        param_abs_mean = torch.abs(param).mean().item()
                        param_std = torch.std(param).item()
                        logger.error(f"  Parameter stats - Abs mean: {param_abs_mean:.6e}, Std: {param_std:.6e}")
                    
                    # Get stats of finite gradient values
                    grad_finite = param.grad[torch.isfinite(param.grad)]
                    if grad_finite.numel() > 0:
                        grad_abs_mean = torch.abs(grad_finite).mean().item()
                        grad_max = torch.abs(grad_finite).max().item()
                        logger.error(f"  Gradient stats - Abs mean: {grad_abs_mean:.6e}, Max abs: {grad_max:.6e}")
                
                has_nan = True
                
                if self.dump_count < self.max_dumps:
                    dump_file = self.save_dir / f"grad_{name}_nan_{time.strftime('%Y%m%d-%H%M%S')}.pt"
                    if param.grad is not None:
                        torch.save(param.grad.detach().cpu(), dump_file)
                    self.dump_count += 1
        
        return not has_nan
    
    def disable(self):
        """Disable the NaN detector hooks to improve performance."""
        self.debug_active = False
        
    def enable(self):
        """Enable the NaN detector hooks."""
        self.debug_active = True
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class BaseTrainingStrategy:
    """
    Base Training Strategy.

    Base class that all training strategies inherit from.
    """

    def __init__(
        self,
        sampler: BaseSampler,
        loss_function: Callable,
        kd_loss: Callable | None = None,
        device: str = "cuda",
        lora: bool = False,
        fabric: Fabric | None = None,
        grad_clip_value: float = 1.0,  # Added grad clip value
        debug_nan: bool = False,  # Enable NaN debugging
        nan_dump_dir: str = "nan_debug",  # Directory for NaN dumps
        max_nan_dumps: int = 3,  # Max number of dumps to save
        **kwargs,
    ):
        """
        Initialises a `BaseTrainingStrategy`
        Args:
            sampler: sampler that returns a sub-network when called
            loss_function: loss function to compute the loss of a sub-network
            device: device to run the model on
            grad_clip_value: value to clip gradients at
            debug_nan: whether to enable NaN debugging
            nan_dump_dir: directory to save NaN dumps
            max_nan_dumps: maximum number of NaN dumps to save
            **kwargs:
        """
        self.sampler = sampler
        self.loss_function = loss_function
        self.device = device
        self.kd_loss = kd_loss
        self.lora = lora
        self.fabric = fabric
        self.grad_clip_value = grad_clip_value
        self.debug_nan = debug_nan
        self.nan_dump_dir = Path(nan_dump_dir)
        self.max_nan_dumps = max_nan_dumps
        self.nan_detector = None
        self.loss_history = []
        self.dump_count = 0
        if isinstance(self.kd_loss, DistillLoss):
            if not isinstance(loss_function, torch.nn.CrossEntropyLoss):
                raise TypeError(
                    "KD Loss not yet supported: Expected torch.nn.CrossEntropyLoss"
                )

    def chunked_loss(self, model, inputs, y):
        y_hat = model(inputs, lm_head_chunk_size=128)
        # Safety check for tensor shapes
        if y_hat[-1].shape[:-2] != y[..., 1:].shape[:-1]:
            # Log the shape mismatch for debugging
            print(f"Shape mismatch: y_hat[-1]={y_hat[-1].shape}, y[..., 1:]={y[..., 1:].shape}")
            # Adjust shapes if needed 
            # (this is a placeholder - adjust according to your model's actual requirements)
            y_hat[-1] = y_hat[-1][..., :y[..., 1:].shape[-2], :]
        return self.loss_function(y_hat, y[..., 1:])

    def compute_loss(self, model, inputs, outputs):
        """
        Compute the loss with safety checks for NaN/Inf values.
        """
        # Inicializa o detector de NaN se necessário
        if self.debug_nan:
            self._setup_nan_detector(model)
            
        # Verifica se há NaN nos inputs
        if self.debug_nan and self.nan_detector:
            input_ok = self.check_batch({k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)})
            if not input_ok:
                logger.warning("NaN detected in inputs - attempting to fix")
                # Tenta corrigir inputs com NaN/Inf
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                        mask = ~torch.isfinite(v)
                        if mask.any():
                            logger.info(f"Replacing {mask.sum().item()} NaN/Inf values in input '{k}' with zeros")
                            inputs[k] = v.clone()
                            inputs[k][mask] = 0.0
        
        # Use detecção de anomalias no autograd para melhor diagnóstico
        with torch.autograd.set_detect_anomaly(True):
            try:
                # Forward pass
                if self.lora:
                    # Usa chunked_loss para LoRA
                    y_hat = model(inputs, lm_head_chunk_size=128)
                    
                    # Verifica compatibilidade de formatos
                    if y_hat[-1].shape[:-2] != outputs[..., 1:].shape[:-1]:
                        logger.warning(f"Shape mismatch: y_hat[-1]={y_hat[-1].shape}, outputs[..., 1:]={outputs[..., 1:].shape}")
                        # Ajusta formatos se necessário
                        y_hat[-1] = y_hat[-1][..., :outputs[..., 1:].shape[-2], :]
                    
                    loss = self.loss_function(y_hat, outputs[..., 1:])
                else:
                    # Forward pass padrão
                    y_hat = model(**inputs)
                    
                    # Verifica NaN nos outputs do modelo
                    if hasattr(y_hat, "logits"):
                        logits = y_hat.logits
                        if not torch.isfinite(logits).all():
                            logger.warning("NaN/Inf detected in logits - fixing")
                            # Clone e corrige logits
                            mask = ~torch.isfinite(logits)
                            if mask.any():
                                logger.info(f"Replacing {mask.sum().item()} NaN/Inf values in logits with zeros")
                                fixed_logits = logits.clone()
                                fixed_logits[mask] = 0.0
                                # Substitui no objeto original
                                y_hat.logits = fixed_logits
                                
                                # Salva estado do modelo para debug
                                if self.debug_nan:
                                    self.dump_state(model, "nan_in_logits")
                    
                    # Compute loss
                    loss = self.loss_function(y_hat, outputs)
                
                # Verifica se o loss é finito
                if not torch.isfinite(loss).all():
                    logger.warning(f"NaN/Inf detected in loss: {loss.item() if loss.numel() == 1 else loss}")
                    # Salva estado do modelo para debug
                    if self.debug_nan:
                        self.dump_state(model, "nan_in_loss")
                    # Usa um loss padrão para evitar quebrar o treinamento
                    loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
                else:
                    # Registra loss no histórico se for finito
                    self.loss_history.append(loss.item() if loss.numel() == 1 else loss.mean().item())
                    
                    # Analisa possíveis picos no loss (indicativo de instabilidade)
                    if len(self.loss_history) > 20:
                        spikes = self.analyze_loss_spikes(window=10, threshold=5.0)
                        if spikes:
                            logger.warning(f"Loss spikes detected at steps {spikes} - possible instability")
                
                return loss
                
            except RuntimeError as e:
                # Captura erros durante o forward/backward
                logger.error(f"Runtime error in compute_loss: {str(e)}")
                # Salva estado do modelo para debug
                if self.debug_nan:
                    self.dump_state(model, "runtime_error")
                # Retorna um loss padrão para não quebrar o treinamento
                return torch.tensor(1.0, device=self.device, requires_grad=True)

    def _setup_nan_detector(self, model):
        """Initialize NaN detection capabilities for the model."""
        if self.debug_nan and self.nan_detector is None:
            self.nan_detector = self._create_nan_detector(model)
            logger.info("NaN detector initialized")
        
    def _create_nan_detector(self, model):
        """Create a NaN detector for the model."""
        return NaNDetector(model, self.nan_dump_dir, self.max_nan_dumps)
        
    def _cleanup_nan_detector(self):
        """Remove NaN detector hooks."""
        if self.nan_detector is not None:
            self.nan_detector.remove_hooks()
            self.nan_detector = None
    
    def check_batch(self, batch, prefix="batch"):
        """Check an input batch for NaN/Inf values."""
        if not self.debug_nan or self.nan_detector is None:
            return True
        return self.nan_detector.check_batch(batch, prefix)
    
    def check_loss(self, loss):
        """Check if loss is NaN/Inf and log detailed information."""
        if not self.debug_nan or self.nan_detector is None:
            return torch.isfinite(loss).all() if isinstance(loss, torch.Tensor) else True
        return self.nan_detector.check_loss(loss)
    
    def check_model_parameters(self, model):
        """Check all model parameters for NaN/Inf values."""
        if not self.debug_nan or self.nan_detector is None:
            return self._check_model_parameters_basic(model)
        return self.nan_detector.check_model_parameters()
    
    def _check_model_parameters_basic(self, model):
        """Basic check for NaN/Inf in model parameters."""
        has_nan = False
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                logger.warning(f"Parameter '{name}' has {nan_count} NaN and {inf_count} Inf values")
                has_nan = True
        return not has_nan
    
    def check_gradients(self, model):
        """Check all model gradients for NaN/Inf values."""
        if not self.debug_nan or self.nan_detector is None:
            return self._check_gradients_basic(model)
        return self.nan_detector.check_gradients()
    
    def _check_gradients_basic(self, model):
        """Basic check for NaN/Inf in gradients."""
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                nan_count = torch.isnan(param.grad).sum().item()
                inf_count = torch.isinf(param.grad).sum().item()
                logger.warning(f"Gradient for '{name}' has {nan_count} NaN and {inf_count} Inf values")
                has_nan = True
        return not has_nan
        
    def fix_nan_parameters(self, model):
        """Replace NaN/Inf values in model parameters with zeros."""
        fixed_count = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                nan_mask = ~torch.isfinite(param)
                if nan_mask.any():
                    nan_count = nan_mask.sum().item()
                    logger.warning(f"Fixing {nan_count} NaN/Inf values in parameter {name}")
                    param.data[nan_mask] = 0.0
                    fixed_count += nan_count
        return fixed_count

    def fix_nan_gradients(self, model):
        """Replace NaN/Inf values in model gradients with zeros."""
        fixed_count = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    nan_mask = ~torch.isfinite(param.grad)
                    if nan_mask.any():
                        nan_count = nan_mask.sum().item()
                        logger.warning(f"Fixing {nan_count} NaN/Inf values in gradient for {name}")
                        param.grad.data[nan_mask] = 0.0
                        fixed_count += nan_count
        return fixed_count
    
    def analyze_loss_spikes(self, window=10, threshold=5.0):
        """Analyze loss values for suspicious spikes that might precede NaN values."""
        if len(self.loss_history) < window + 1:
            return []
        
        spikes = []
        for i in range(window, len(self.loss_history)):
            prev_avg = sum(self.loss_history[i-window:i]) / window
            if prev_avg > 0 and self.loss_history[i] > prev_avg * threshold:
                spikes.append(i)
        
        return spikes
    
    def initialize_model_safely(self, model):
        """Initialize model weights to prevent NaN issues."""
        for name, module in model.named_modules():
            # Linear layers
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight.data)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)
            
            # LayerNorm (common in transformer models)
            elif "LayerNorm" in module.__class__.__name__:
                if hasattr(module, 'weight'):
                    torch.nn.init.ones_(module.weight.data)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)
            
            # Embedding layers
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
    
    def print_model_parameter_stats(self, model):
        """Print statistics about model parameters to help diagnose issues."""
        logger.info("Model Parameter Statistics:")
        
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Skip printing for very large parameters to avoid flooding logs
                if param.numel() > 1000000:  # Skip detailed stats for params > 1M elements
                    logger.info(f"  {name}: shape={tuple(param.shape)}, elements={param.numel()/1e6:.2f}M")
                    if torch.isfinite(param).all():
                        logger.info(f"    min={param.min().item():.6e}, max={param.max().item():.6e}, mean={param.mean().item():.6e}")
                    else:
                        nan_count = torch.isnan(param).sum().item()
                        inf_count = torch.isinf(param).sum().item()
                        logger.warning(f"    contains {nan_count} NaN and {inf_count} Inf values!")
                total_params += param.numel()
        
        logger.info(f"Total trainable parameters: {total_params/1e6:.2f}M")
    
    def dump_state(self, model, prefix):
        """Dump model state for debugging when NaN is detected."""
        if self.dump_count >= self.max_nan_dumps:
            return
            
        self.nan_dump_dir.mkdir(exist_ok=True, parents=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dump_file = self.nan_dump_dir / f"{prefix}_{timestamp}.pt"
        
        # Create a state dictionary with model parameters and their gradients
        state = {
            'parameters': {},
            'gradients': {},
            'timestamp': timestamp
        }
        
        for name, param in model.named_parameters():
            state['parameters'][name] = param.detach().cpu()
            if param.grad is not None:
                state['gradients'][name] = param.grad.detach().cpu()
        
        torch.save(state, dump_file)
        logger.warning(f"Dumped model state to {dump_file}")
        self.dump_count += 1

    def __call__(self, model, inputs, outputs, scale_loss=1, **kwargs):
        raise NotImplementedError
