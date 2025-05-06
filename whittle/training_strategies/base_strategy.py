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
        **kwargs,
    ):
        """
        Initialises a `BaseTrainingStrategy`
        Args:
            sampler: sampler that returns a sub-network when called
            loss_function: loss function to compute the loss of a sub-network
            device: device to run the model on
            **kwargs:
        """
        self.sampler = sampler
        self.loss_function = loss_function
        self.device = device
        self.kd_loss = kd_loss
        self.lora = lora
        self.fabric = fabric
        if isinstance(self.kd_loss, DistillLoss):
            if not isinstance(loss_function, torch.nn.CrossEntropyLoss):
                raise TypeError(
                    "KD Loss not yet supported: Expected torch.nn.CrossEntropyLoss"
                )

    def chunked_loss(self, model, inputs, y):
        y_hat = model(inputs, lm_head_chunk_size=128)
        y_hat[-1] = y_hat[-1][..., :-1, :]
        return self.loss_function(y_hat, y[..., 1:])

    def compute_loss(self, model, inputs, outputs):
        # Debug: checar inputs/outputs
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                print(f"[DEBUG compute_loss] inputs['{k}'] contém NaN/Inf")
        if isinstance(outputs, torch.Tensor) and not torch.isfinite(outputs).all():
            print(f"[DEBUG compute_loss] outputs contém NaN/Inf")

        # Ativa detect_anomaly para rastrear o op exato
        with torch.autograd.set_detect_anomaly(True):
            if self.lora:
                loss = self.chunked_loss(model, inputs, outputs)
            else:
                y_hat = model(**inputs)
                # Debug: checar y_hat
                if hasattr(y_hat, "logits"):
                    logits = y_hat.logits
                    if not torch.isfinite(logits).all():
                        print(f"[DEBUG compute_loss] logits contém NaN/Inf (min={logits.min()}, max={logits.max()})")
                elif isinstance(y_hat, torch.Tensor):
                    if not torch.isfinite(y_hat).all():
                        print(f"[DEBUG compute_loss] y_hat tensor contém NaN/Inf")

                loss = self.loss_function(y_hat, outputs)
                # Debug: checar loss
                if not torch.isfinite(loss):
                    print(f"[DEBUG compute_loss] loss é NaN/Inf -> {loss}")

        return loss

    def __call__(self, model, inputs, outputs, scale_loss=1, **kwargs):
        raise NotImplementedError
