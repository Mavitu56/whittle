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

        return self

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