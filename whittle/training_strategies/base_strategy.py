from __future__ import annotations

from collections.abc import Callable

import torch
from lightning.fabric import Fabric

from whittle.loss import DistillLoss
from whittle.sampling.random_sampler import BaseSampler


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
        gradient_clip_value: float = 1.0,
        **kwargs,
    ):
        """
        Initialises a `BaseTrainingStrategy`
        Args:
            sampler: sampler that returns a sub-network when called
            loss_function: loss function to compute the loss of a sub-network
            kd_loss: knowledge distillation loss function
            device: device to run the model on
            lora: whether to use LoRA
            fabric: Lightning Fabric instance for distributed training
            gradient_clip_value: maximum gradient norm for gradient clipping (default: 1.0)
            **kwargs: Additional keyword arguments
        """
        self.sampler = sampler
        self.loss_function = loss_function
        self.device = device
        self.kd_loss = kd_loss
        self.lora = lora
        self.fabric = fabric
        self.gradient_clip_value = gradient_clip_value
        
        if isinstance(self.kd_loss, DistillLoss):
            if not isinstance(loss_function, torch.nn.CrossEntropyLoss):
                raise TypeError(
                    "KD Loss not yet supported: Expected torch.nn.CrossEntropyLoss"
                )
    
    def check_for_nan_inf(self, tensor, name="tensor"):
        """
        Verifica se um tensor contém valores NaN ou Inf.
        
        Args:
            tensor: O tensor a ser verificado
            name: Nome identificador para o tensor em mensagens de erro
            
        Returns:
            bool: True se contém NaN/Inf, False caso contrário
        """
        if tensor is None:
            return False
        
        has_nan = torch.isnan(tensor).any().item() if tensor.numel() > 0 else False
        has_inf = torch.isinf(tensor).any().item() if tensor.numel() > 0 else False
        
        if has_nan or has_inf:
            print(f"ALERTA: Detectado {'NaN' if has_nan else ''} {'Inf' if has_inf else ''} em {name}")
            return True
        return False

    def apply_gradient_clipping(self, model):
        """
        Aplica gradient clipping nas variáveis com gradiente.
        
        Args:
            model: O modelo cujos gradientes serão limitados
        """
        if self.gradient_clip_value > 0 and self.fabric is None:
            # Verifica se há gradientes antes de aplicar clipping
            has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            if has_grads:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad and p.grad is not None], 
                    self.gradient_clip_value
                )

    def chunked_loss(self, model, inputs, y):
        y_hat = model(inputs, lm_head_chunk_size=128)
        y_hat[-1] = y_hat[-1][..., :-1, :]
        loss = self.loss_function(y_hat, y[..., 1:])
        
        # Verificar se a loss é válida
        self.check_for_nan_inf(loss, name="chunked_loss")
        
        return loss

    def compute_loss(self, model, inputs, outputs):
        try:
            if self.lora:
                loss = self.chunked_loss(model, inputs, outputs)
            else:
                y_hat = model(**inputs)
                loss = self.loss_function(y_hat, outputs)
                
            # Verificar se a loss computada é válida
            self.check_for_nan_inf(loss, name="compute_loss")
            
            return loss
        except RuntimeError as e:
            print(f"Erro ao computar loss: {str(e)}")
            # Retornar um tensor zero como fallback para evitar quebra total do treinamento
            # Isso permitirá que o treinamento continue, mas o otimizador não atualizará os pesos
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def __call__(self, model, inputs, outputs, scale_loss=1, **kwargs):
        raise NotImplementedError
