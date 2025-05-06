class LLaMASuperNetMixin:
    search_space = None
    handles = None
    _original_embedding_backup = None  # Backup da matriz de embedding original

    def select_sub_network(self, sub_network_config=None, **kwargs):
        # Handle both a config dict or keyword arguments
        config = sub_network_config if sub_network_config is not None else kwargs

        # Verificar e recuperar matriz de embeddings se necessário
        self._ensure_valid_embeddings()

        # Debug: checar embeddings após weight sharing
        import torch
        if hasattr(self, 'model') and hasattr(self.model, 'embed_tokens'):
            emb = self.model.embed_tokens.weight
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                print(f"[DEBUG] NaN/Inf detectado em embeddings após select_sub_network")

        head_mask, ffn_mask = self.search_space.config_to_mask(config)
        head_mask = head_mask.to(device="cuda", dtype=self.dtype)
        ffn_mask = ffn_mask.to(device="cuda", dtype=self.dtype)
        self.handles = mask_llama(self, ffn_mask, head_mask)
        return self

    def set_sub_network(self, sub_network_config=None, **kwargs):
        # Alias for select_sub_network to maintain compatibility
        return self.select_sub_network(sub_network_config, **kwargs)

    def reset_super_network(self):
        if self.handles:
            for handle in self.handles:
                handle.remove()

        # Verificar e recuperar matriz de embeddings se necessário
        self._ensure_valid_embeddings()

        return self

    def _ensure_valid_embeddings(self):
        """
        Verifica se a matriz de embeddings possui valores NaN ou Inf e
        recupera/reinicializa se necessário.

        Esta é uma função de recuperação de emergência que impede que NaNs
        persistam no modelo e causem problemas em iterações futuras.
        """
        import torch

        if not hasattr(self, 'model') or not hasattr(self.model, 'embed_tokens'):
            return

        embeddings = self.model.embed_tokens.weight

        # Verificar se há NaNs ou Infs nos embeddings
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            print(f"[DEBUG] NaN/Inf detectado em embeddings em _ensure_valid_embeddings")

            # Calcular quantos valores inválidos temos
            nan_count = torch.isnan(embeddings).sum().item()
            inf_count = torch.isinf(embeddings).sum().item()
            total = embeddings.numel()

            print(f"[WARNING] Detectados {nan_count} NaNs e {inf_count} Infs na matriz de embeddings " +
                  f"({(nan_count + inf_count) / total * 100:.2f}% do total).")
            print("[INFO] Aplicando recuperação de emergência na matriz de embeddings.")

            # Se temos um backup, usar para recuperar
            if self._original_embedding_backup is not None:
                print("[INFO] Restaurando matriz de embeddings do backup.")
                # Clonar para evitar problemas de referência
                self.model.embed_tokens.weight.data.copy_(self._original_embedding_backup)
            else:
                # Se não temos backup, precisamos reinicializar
                print("[INFO] Reinicializando matriz de embeddings (sem backup disponível).")
                # Determinar o método de inicialização apropriado
                vocab_size, embedding_dim = embeddings.shape

                # Reinicializar com distribuição normal pequena
                # Similar ao usado pelo LLaMA original
                std = 0.02  # Valor padrão típico para embeddings
                new_embeddings = torch.normal(
                    mean=0.0, std=std,
                    size=(vocab_size, embedding_dim),
                    device=embeddings.device,
                    dtype=embeddings.dtype
                )

                # Aplicar a nova matriz de embeddings
                self.model.embed_tokens.weight.data.copy_(new_embeddings)

                # Criar backup agora
                self._original_embedding_backup = new_embeddings.clone()

            # Verificar se a recuperação funcionou
            if torch.isnan(self.model.embed_tokens.weight).any():
                print("[ERROR] Falha na recuperação da matriz de embeddings!")
            else:
                print("[INFO] Matriz de embeddings recuperada com sucesso.")
        else:
            # Se os embeddings são válidos e não temos backup, criar um agora
            if self._original_embedding_backup is None:
                # Criar backup apenas na primeira vez que vemos embeddings válidos
                self._original_embedding_backup = embeddings.clone()
                print("[INFO] Backup inicial da matriz de embeddings criado.")

    def adaptive_scale_for_config(self, config):
        """
        Calcula um fator de escala apropriado para os gradientes baseado na configuração.

        Args:
            config: Dicionário de configuração da rede (num_layers, num_units, num_heads)

        Returns:
            float: Fator de escala entre 0.1 e 1.0
        """
        # Definição de limites para configurações extremas
        min_scale = 0.1  # Fator de escala mínimo
        thresholds = {
            'num_layers': 1,
            'num_units': 32,
            'num_heads': 1
        }

        # Se não temos configuração, retornar escala normal
        if not config:
            return 1.0

        # Calcular fatores de escala para cada dimensão
        scale_factors = []

        for dim, threshold in thresholds.items():
            if dim in config:
                # Quanto menor o valor em relação ao limite, menor o fator
                dim_scale = min(1.0, max(min_scale, config[dim] / max(1, threshold)))
                scale_factors.append(dim_scale)

        # Se não temos fatores, retornar escala normal
        if not scale_factors:
            return 1.0

        # Retornar média dos fatores
        return sum(scale_factors) / len(scale_factors)


class LLaMASuperNetMixinLAYERSpace(LLaMASuperNetMixin):
    @property
    def search_space(self):
        return LayerSearchSpace(self.config)


class LLaMASuperNetMixinMEDIUMSpace(LLaMASuperNetMixin):
    @property
    def search_space(self):
        return MediumSearchSpace(self.config)


class LLaMASuperNetMixinLARGESpace(LLaMASuperNetMixin):
    @property
    def search_space(self):
        return FullSearchSpace(self.config)


class LLaMASuperNetMixinSMALLSpace(LLaMASuperNetMixin):
    @property
    def search_space(self):
        return SmallSearchSpace(self.config)


class SuperNetLlamaForCausalLMSMALL(
    LlamaForCausalLM, LLaMASuperNetMixinSMALLSpace
):
    def forward(self, inputs=None, **kwargs):
        # Verificar e corrigir embeddings antes de processar o forward
        self._ensure_valid_embeddings()

        if inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForSequenceClassificationSMALL(
    LlamaForSequenceClassification, LLaMASuperNetMixinSMALLSpace
):
    def forward(self, inputs=None, **kwargs):
        # Verificar e corrigir embeddings antes de processar o forward
        self._ensure_valid_embeddings()

        if inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForCausalLMLAYER(
    LlamaForCausalLM, LLaMASuperNetMixinLAYERSpace
):
    def forward(self, inputs=None, **kwargs):
        # Verificar e corrigir embeddings antes de processar o forward
        self._ensure_valid_embeddings()

        if inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForSequenceClassificationLAYER(
    LlamaForSequenceClassification, LLaMASuperNetMixinLAYERSpace
):
    def forward(self, inputs=None, **kwargs):
        # Verificar e corrigir embeddings antes de processar o forward
        self._ensure_valid_embeddings()

        if inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForCausalLMMEDIUM(
    LlamaForCausalLM, LLaMASuperNetMixinMEDIUMSpace
):
    def forward(self, inputs=None, **kwargs):
        # Verificar e corrigir embeddings antes de processar o forward
        self._ensure_valid_embeddings()

        if inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForSequenceClassificationMEDIUM(
    LlamaForSequenceClassification, LLaMASuperNetMixinMEDIUMSpace
):
    def forward(self, inputs=None, **kwargs):
        # Verificar e corrigir embeddings antes de processar o forward
        self._ensure_valid_embeddings()

        if inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForCausalLMLARGE(
    LlamaForCausalLM, LLaMASuperNetMixinLARGESpace
):
    def forward(self, inputs=None, **kwargs):
        # Verificar e corrigir embeddings antes de processar o forward
        self._ensure_valid_embeddings()

        if inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForSequenceClassificationLARGE(
    LlamaForSequenceClassification, LLaMASuperNetMixinLARGESpace
):
    def forward(self, inputs=None, **kwargs):
        # Verificar e corrigir embeddings antes de processar o forward
        self._ensure_valid_embeddings()

        if inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)