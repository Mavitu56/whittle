from __future__ import annotations

from typing import Any
import torch

from whittle.training_strategies.base_strategy import BaseTrainingStrategy


class SandwichStrategy(BaseTrainingStrategy):
    """
    Sandwich strategy.

    In each step, the sandwich strategy updates the super-network, the smallest, and a set of randomly sampled
    sub-networks.

    refs:
        Universally Slimmable Networks and Improved Training Techniques
        Jiahui Yu, Thomas Huang
        International Conference on Computer Vision 2019
        https://arxiv.org/abs/1903.05134
    """

    def __init__(self, random_samples: int = 2, gradient_clip_value: float = 1.0, 
                 enable_gradient_accumulation: bool = False, **kwargs: Any):
        """
        Initialises a `SandwichStrategy`

        Args:
            random_samples: the number of randomly sampled sub-networks to sample and update in each step
            gradient_clip_value: valor máximo para clipping de gradientes (default: 1.0)
            enable_gradient_accumulation: se True, acumula gradientes em vez de realizar backward separado (mais estável)
            **kwargs: kwargs of `BaseTrainingStrategy`
        """
        super().__init__(**kwargs)
        self.random_samples = random_samples
        self.gradient_clip_value = gradient_clip_value
        self.enable_gradient_accumulation = enable_gradient_accumulation
        
    def _check_for_nan_inf(self, loss, name="loss"):
        """Verifica se a loss tem valores NaN ou Inf e registra informação"""
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"ALERTA: Detectado NaN/Inf em {name}: {loss.item()}")
            return True
        return False

    def __call__(self, model, inputs, outputs, scale_loss=1, **kwargs):
        # Implementação com acumulação de gradientes para maior estabilidade numérica
        if self.enable_gradient_accumulation:
            return self._call_with_grad_accumulation(model, inputs, outputs, scale_loss, **kwargs)
        else:
            return self._call_original(model, inputs, outputs, scale_loss, **kwargs)
            
    def _call_with_grad_accumulation(self, model, inputs, outputs, scale_loss=1, **kwargs):
        """
        Versão da estratégia Sandwich com acumulação de gradientes para maior estabilidade.
        Acumula gradientes de todas as configurações de rede e depois aplica uma única vez.
        """
        total_loss = 0
        networks = []
        
        # Preparar a lista de redes para treinar: super-rede, redes aleatórias e menor rede
        networks.append(("super", lambda: model.reset_super_network()))
        
        for i in range(self.random_samples):
            config = self.sampler.sample()
            networks.append((f"random_{i}", lambda c=config: model.set_sub_network(**c)))
            
        smallest_config = self.sampler.get_smallest_sub_network()
        networks.append(("smallest", lambda: model.set_sub_network(**smallest_config)))
        
        # Zerar gradientes antes de iniciar
        if self.fabric is None and hasattr(model, 'zero_grad'):
            model.zero_grad(set_to_none=True)  # set_to_none=True é mais eficiente
        
        # Processar cada configuração de rede
        for net_name, set_net_fn in networks:
            # Configurar a rede atual
            set_net_fn()
            
            # Calcular a loss da configuração atual
            try:
                with torch.cuda.amp.autocast(enabled=False):  # Desativar autocast para prevenir instabilidades numéricas
                    loss = self.compute_loss(model, inputs, outputs)
                    
                # Verificar se a loss tem valores inválidos
                if self._check_for_nan_inf(loss, name=f"loss_{net_name}"):
                    # Se detectar NaN, pule esta iteração e continue com a próxima rede
                    print(f"Pulando backward para {net_name} devido a NaN/Inf")
                    continue
                    
                # Aplicar scaling e backward
                scaled_loss = loss * (scale_loss / len(networks))
                if self.fabric is None:
                    scaled_loss.backward()
                else:
                    self.fabric.backward(scaled_loss)
                    
                total_loss += loss.item()
                
            except RuntimeError as e:
                print(f"Erro durante o processamento da rede {net_name}: {str(e)}")
                continue
        
        # Aplicar gradient clipping após acumular todos os gradientes
        if self.gradient_clip_value > 0:
            if self.fabric is None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.gradient_clip_value
                )
        
        # Resetar para supernetwork para consistência
        model.reset_super_network()
        return loss,total_loss

    def _call_original(self, model, inputs, outputs, scale_loss=1, **kwargs):
        """
        Implementação original da estratégia Sandwich, com adição de verificação de NaN
        e gradient clipping para maior estabilidade.
        """
        total_loss = 0
        
        # update super-network
        model.reset_super_network()
        loss = self.compute_loss(model, inputs, outputs)
        
        # Verificar NaN/Inf
        if not self._check_for_nan_inf(loss, "super_network_loss"):
            loss *= scale_loss
            loss.backward() if self.fabric is None else self.fabric.backward(loss)
            
            # Aplicar gradient clipping após cada backward
            if self.gradient_clip_value > 0 and self.fabric is None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.gradient_clip_value
                )
                
            total_loss += loss.item()
            
        # update random sub-networks
        for i in range(self.random_samples):
            # Zera gradientes entre atualizações de redes diferentes para evitar acúmulo
            if self.fabric is None and hasattr(model, 'zero_grad'):
                model.zero_grad(set_to_none=True)
                
            config = self.sampler.sample()
            model.set_sub_network(**config)
            loss = self.compute_loss(model, inputs, outputs)

            # Verificar NaN/Inf
            if not self._check_for_nan_inf(loss, f"random_network_{i}_loss"):
                loss *= scale_loss
                loss.backward() if self.fabric is None else self.fabric.backward(loss)
                
                # Aplicar gradient clipping após cada backward
                if self.gradient_clip_value > 0 and self.fabric is None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.gradient_clip_value
                    )
                    
                model.reset_super_network()
                total_loss += loss.item()

        # Zera gradientes antes da menor rede
        if self.fabric is None and hasattr(model, 'zero_grad'):
            model.zero_grad(set_to_none=True)
            
        # smallest network
        config = self.sampler.get_smallest_sub_network()
        model.set_sub_network(**config)
        loss = self.compute_loss(model, inputs, outputs)
        
        # Verificar NaN/Inf
        if not self._check_for_nan_inf(loss, "smallest_network_loss"):
            loss *= scale_loss
            loss.backward() if self.fabric is None else self.fabric.backward(loss)
            
            # Aplicar gradient clipping após cada backward
            if self.gradient_clip_value > 0 and self.fabric is None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.gradient_clip_value
                )
                
            model.reset_super_network()
            total_loss += loss.item()

        return total_loss
