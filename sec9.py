import logging

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding, default_data_collator, DataCollatorForSeq2Seq


logger = logging.getLogger(__name__)


class DataWrapper:
    def __init__(self, training_args, model_args, data_args):
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.model_type = model_args.model_name_or_path

        # Load tokenizer
        self.tokenizer = self.get_tokenizer()

        # Configurações específicas para modelos diferentes
        if (
            self.model_type.startswith("gpt2")
            or "pythia" in self.model_type
            or self.model_type.startswith("distilgpt2")
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Configuração específica para LLaMA
        elif "llama" in self.model_type.lower():
            # Para LLaMA 3, o pad token é o token </s> (EOS)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Definindo pad_token = eos_token para o modelo LLaMA: {self.tokenizer.pad_token}")

        # Padding strategy
        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Determine max_seq_length
        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(
            self.data_args.max_seq_length, self.tokenizer.model_max_length
        )

        self.train_data, self.valid_data, self.test_data = self._load_data()

        data_collator = self.get_data_collator()

        self.train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.training_args.per_device_train_batch_size,
            collate_fn=data_collator,
            shuffle=True,
        )

        self.eval_dataloader = DataLoader(
            self.valid_data,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )

        self.test_dataloader = DataLoader(
            self.test_data,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )

        self.num_labels = self.get_num_labels(self.data_args)

    def get_num_labels(self, data_args):
        if data_args.is_regression:
            num_labels = 1
        elif hasattr(self.train_data.features, 'get') and self.train_data.features.get("label") is not None:
            label_list = self.train_data.features["label"].names
            num_labels = len(label_list)
        else:
            # Para modelos de geração como o LLaMA com Alpaca
            num_labels = None
        return num_labels

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_type,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
            trust_remote_code=True,  # Necessário para alguns modelos como LLaMA
        )

    def get_data_loaders(self):
        # Debug: checar batches para NaN/Inf
        import torch
        def debug_batch(batch, loader_name):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                    print(f"[DEBUG] NaN/Inf detectado em {loader_name}['{k}']")
        # Testa um batch de cada loader
        for loader, name in zip([self.train_dataloader, self.eval_dataloader, self.test_dataloader],
                                ['train', 'eval', 'test']):
            try:
                batch = next(iter(loader))
                debug_batch(batch, name)
            except Exception as e:
                print(f"[DEBUG] Erro ao checar batch {name}: {e}")
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_collator(self):
        if "llama" in self.model_type.lower() and hasattr(self.data_args, 'task_name') and self.data_args.task_name == 'alpaca':
            # Para LLaMA com tarefas de geração, usamos o DataCollatorForSeq2Seq
            return DataCollatorForSeq2Seq(
                self.tokenizer,
                model=None,  # Modelo não é necessário para collation
                padding='max_length',  # Pad to max_length to avoid warning
                max_length=self.max_seq_length,
                pad_to_multiple_of=8 if self.training_args.fp16 else None,
                return_tensors="pt"  # Garantir que retorne tensores PyTorch
            )
        elif self.data_args.pad_to_max_length:
            # Se já tivermos feito padding para max_length, usamos o collator padrão
            return default_data_collator
        else:
            # Caso contrário, usamos o DataCollatorWithPadding para padding dinâmico
            return DataCollatorWithPadding(
                self.tokenizer,
                padding=True,
                pad_to_multiple_of=8 if self.training_args.fp16 else None
            )

    def _load_data(self):
        pass