logger = logging.getLogger(__name__)

class Alpaca(DataWrapper):
    def _load_data(self):
        """
        Carrega o dataset Alpaca a partir de um arquivo JSON.
        Se o arquivo não for especificado ou não existir, baixa automaticamente.
        """
        # URL para o dataset Alpaca oficial
        ALPACA_DATASET_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

        # Verifica se o arquivo foi fornecido e existe
        file_path = self.data_args.train_file
        if not file_path or not os.path.exists(file_path):
            logger.info(f"Arquivo de treinamento não encontrado: {file_path}")
            logger.info(f"Baixando dataset Alpaca de {ALPACA_DATASET_URL}")

            # Cria o diretório cache se não existir
            cache_dir = self.model_args.cache_dir
            if not cache_dir:
                cache_dir = os.path.join(os.getcwd(), "cache")
            os.makedirs(cache_dir, exist_ok=True)

            # Baixa o dataset
            file_path = os.path.join(cache_dir, "alpaca_data.json")
            if not os.path.exists(file_path):
                response = requests.get(ALPACA_DATASET_URL)
                response.raise_for_status()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"Dataset Alpaca baixado para {file_path}")

        # Carrega os dados do arquivo
        logger.info(f"Carregando dataset Alpaca de {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        # Debug: checar se há NaN/Inf nos dados brutos
        import torch
        for i, item in enumerate(raw_data):
            if any(isinstance(v, float) and (torch.isnan(torch.tensor(v)) or torch.isinf(torch.tensor(v))) for v in item.values()):
                print(f"[DEBUG] NaN/Inf detectado em raw_data[{i}]: {item}")

        # Convertendo para o formato datasets
        alpaca_dataset = Dataset.from_dict({
            "instruction": [item["instruction"] for item in raw_data],
            "input": [item.get("input", "") for item in raw_data],
            "output": [item["output"] for item in raw_data]
        })

        # Preprocess função para tokenizar os textos
        def preprocess_function(examples):
            # Formata para o modelo LLaMA
            formatted_inputs = []
            formatted_outputs = []
            filtered_instructions = []
            filtered_inputs = []
            filtered_outputs = []

            # Primeiro tokenizar sem truncamento para verificar comprimento
            for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
                if input_text:
                    prompt = f"<s>[INST] {instruction}\n{input_text} [/INST]"
                else:
                    prompt = f"<s>[INST] {instruction} [/INST]"

                full_text = prompt + output + "</s>"

                # Verificar o comprimento sem aplicar truncamento
                tokens = self.tokenizer(prompt, output + "</s>", truncation=False, add_special_tokens=False)

                # Selecionar apenas exemplos dentro do tamanho máximo
                if len(tokens["input_ids"]) <= self.max_seq_length:
                    filtered_instructions.append(instruction)
                    filtered_inputs.append(input_text)
                    filtered_outputs.append(output)
                    formatted_inputs.append(prompt)
                    formatted_outputs.append(output + "</s>")

            print(f"Filtrando exemplos: {len(formatted_inputs)}/{len(examples['instruction'])} dentro do limite de {self.max_seq_length} tokens")

            # Se não houver exemplos válidos, usar todos com truncamento (fallback)
            if len(formatted_inputs) == 0:
                print(f"Nenhum exemplo dentro do limite de {self.max_seq_length} tokens. Usando todos com truncamento.")
                for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
                    if input_text:
                        prompt = f"<s>[INST] {instruction}\n{input_text} [/INST]"
                    else:
                        prompt = f"<s>[INST] {instruction} [/INST]"
                    formatted_inputs.append(prompt)
                    formatted_outputs.append(output + "</s>")

            # Tokeniza entradas com padding e truncation ativados
            model_inputs = self.tokenizer(
                formatted_inputs,
                padding='max_length',  # Sempre usar 'max_length' quando especificamos max_length
                max_length=self.max_seq_length,
                truncation=True,
                return_tensors="pt"  # Retorna PyTorch tensors
            )
            # Debug: checar NaN/Inf após tokenização
            for k, v in model_inputs.items():
                if not torch.isfinite(v).all():
                    print(f"[DEBUG] NaN/Inf detectado em model_inputs['{k}'] após tokenização")

            # Tokeniza saídas também com padding/truncation
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    formatted_outputs,
                    padding='max_length',  # Sempre usar 'max_length' quando especificamos max_length
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt"  # Retorna PyTorch tensors
                )
            for k, v in labels.items():
                if not torch.isfinite(v).all():
                    print(f"[DEBUG] NaN/Inf detectado em labels['{k}'] após tokenização")

            # Detach tensors para evitar problemas com o graph do PyTorch
            model_inputs = {k: v.detach().clone() for k, v in model_inputs.items()}

            # Cria as labels para o modelo
            model_inputs["labels"] = labels["input_ids"].detach().clone()

            return model_inputs

        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            processed_dataset = alpaca_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not getattr(self.data_args, 'overwrite_cache', False),
                desc="Running tokenizer on dataset",
                remove_columns=["instruction", "input", "output"]  # Remove colunas originais
            )

        # Divide em conjuntos de treino, validação e teste
        split = processed_dataset.train_test_split(
            test_size=0.2, seed=self.data_args.dataset_seed
        )
        train_dataset = split["train"]

        # Divide o conjunto de teste em validação e teste
        subsplit = split["test"].train_test_split(
            test_size=0.5, seed=self.data_args.dataset_seed
        )
        valid_dataset = subsplit["train"]
        test_dataset = subsplit["test"]

        print(f"Dataset carregado: {len(train_dataset)} exemplos de treino, "
                    f"{len(valid_dataset)} de validação, {len(test_dataset)} de teste")

        return train_dataset, valid_dataset, test_dataset

    def get_num_labels(self, data_args):
        # Para modelos de linguagem causal, não usamos classificação com labels
        return None
