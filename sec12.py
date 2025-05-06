# Treinamento NAS para LLaMA usando Alpaca
import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Configuração de logging com mais detalhes e formato melhorado
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # Para saída no console
    ]
)
logger = logging.getLogger(__name__)

# Configuração para mostrar outputs imediatamente (útil em notebooks)
import sys
sys.stdout.flush()

# Função auxiliar para visualização de progresso
def print_status_update(message, flush=True):
    """Imprime uma atualização de status com timestamp."""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}", flush=flush)

def calculate_perplexity(model, dataloader, device):
    """
    Calcula a perplexidade de um modelo em um dataset.
    Específico para modelos de linguagem causal como o LLaMA.
    """
    print_status_update("Iniciando cálculo de perplexidade...")
    model.eval()
    total_loss = 0
    total_length = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 5 == 0:  # Atualização a cada 5 batches
                print_status_update(f"Calculando perplexidade: batch {batch_idx}/{len(dataloader)}")

            batch = {k: v.to(device) for k, v in batch.items()}

            # O modelo calcula a loss automaticamente se fornecer labels
            outputs = model(**batch)

            # Adiciona a loss ponderada pelo número de tokens
            loss = outputs.loss
            total_loss += loss.item() * batch["attention_mask"].sum().item()
            total_length += batch["attention_mask"].sum().item()

    # Perplexidade = exp(loss média)
    avg_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print_status_update(f"Perplexidade calculada: {perplexity:.4f}")
    return {"perplexity": perplexity}

def loss_function(predictions, labels):
    return predictions.loss

def load_checkpoint(checkpoint_dir):
    """
    Verifica e carrega o checkpoint disponível para continuar o treinamento.

    Args:
        checkpoint_dir: Diretório específico do checkpoint a ser carregado

    Returns:
        Um dicionário com o modelo carregado, otimizador, scheduler, scaler,
        o número da próxima época a executar e outras informações do estado
        ou None se não houver checkpoint disponível
    """
    print_status_update(f"Verificando checkpoint em: {checkpoint_dir}")

    checkpoint_dir = "/content/output/pruning_Llama3-1B_alpaca_small_sandwich/checkpoint_epoca_0"

    # Verificar se o diretório existe
    if not os.path.exists(checkpoint_dir) or not os.path.isdir(checkpoint_dir):
        print_status_update(f"Diretório de checkpoint não encontrado: {checkpoint_dir}")
        return None

    # Verificar arquivos necessários
    training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
    optimizer_state_path = os.path.join(checkpoint_dir, "optimizer.pt")
    scheduler_state_path = os.path.join(checkpoint_dir, "scheduler.pt")

    if not all(os.path.exists(p) for p in [training_state_path, optimizer_state_path, scheduler_state_path]):
        print_status_update("Arquivos de checkpoint necessários não encontrados")
        return None

    try:
        from transformers import BitsAndBytesConfig, AutoConfig, get_scheduler, AutoTokenizer
        from torch.optim import AdamW
        from torch.cuda.amp import GradScaler

        # Carregando modelo completo do checkpoint
        print_status_update("Carregando modelo completo do checkpoint...")
        model_config = AutoConfig.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True
        )

        # Selecionar o tipo de modelo baseado no espaço de busca
        model_map = {
            "small": SuperNetLlamaForCausalLMSMALL,
            "medium": SuperNetLlamaForCausalLMMEDIUM,
            "layer": SuperNetLlamaForCausalLMLAYER,
            "large": SuperNetLlamaForCausalLMLARGE
        }
        model_cls = model_map[SEARCH_SPACE]

        # Carregar o modelo com ou sem quantização
        if USE_8BIT_QUANTIZATION:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16  # use float16 compute for faster performance
            )
            model = model_cls.from_pretrained(
                checkpoint_dir,
                quantization_config=bnb_config,
                device_map="auto" if torch.cuda.is_available() and not USE_CPU else None,
                trust_remote_code=True
            )
        else:
            model = model_cls.from_pretrained(checkpoint_dir)

        print_status_update("Modelo carregado com sucesso!")

        # Carregar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            print_status_update(f"Definido pad_token = eos_token para o modelo: {tokenizer.pad_token}")

        # Carregar estado do treinamento
        training_state = torch.load(training_state_path)
        print_status_update(f"Estado do treinamento carregado. Próxima época: {training_state['epoch']}")

        # Restaurar estado RNG para reprodutibilidade
        np.random.set_state(training_state['rng_states']['numpy'])
        torch.set_rng_state(training_state['rng_states']['torch'])
        if training_state['rng_states']['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(training_state['rng_states']['cuda'])

        # Definir otimizador para ser carregado com seu estado mais tarde
        optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.95)
        )

        # Carregar estado do otimizador
        optimizer_state = torch.load(optimizer_state_path)
        optimizer.load_state_dict(optimizer_state)
        print_status_update("Estado do otimizador carregado.")

        # Configurar scheduler
        num_training_steps = int(NUM_EPOCHS * len(train_dataloader) / GRADIENT_ACCUMULATION)
        warmup_steps = int(WARMUP_RATIO * num_training_steps)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        # Carregar estado do scheduler
        scheduler_state = torch.load(scheduler_state_path)
        lr_scheduler.load_state_dict(scheduler_state)
        print_status_update("Estado do scheduler carregado.")

        # Carregar scaler se existir e estiver usando precisão mista
        scaler = None
        if FP16_TRAINING:
            scaler = torch.amp.GradScaler(device_type='cuda' if torch.cuda.is_available() else 'cpu')
            scaler_state_path = os.path.join(checkpoint_dir, "scaler.pt")
            if os.path.exists(scaler_state_path):
                scaler.load_state_dict(torch.load(scaler_state_path))
                print_status_update("Estado do scaler carregado.")

        checkpoint_data = {
            "model": model,
            "tokenizer": tokenizer,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "scaler": scaler,
            "next_epoch": training_state["epoch"],
            "best_perplexity": training_state.get("best_perplexity", float('inf')),
            "total_runtime": training_state.get("total_runtime", 0)
        }

        print_status_update(f"Checkpoint completo carregado com sucesso! Continuando da época {training_state['epoch']}.")
        return checkpoint_data

    except Exception as e:
        print_status_update(f"Erro ao carregar checkpoint: {str(e)}")
        import traceback
        print_status_update(f"Detalhes do erro: {traceback.format_exc()}")
        print_status_update("Iniciando treinamento do zero.")
        return None

def train_llama_nas():
    """Função principal para treinar um modelo LLaMA usando NAS."""
    print_status_update("Configurando o ambiente de treinamento...")

    # Criar diretório de output se não existir
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    print_status_update(f"Diretório de saída: {EXPERIMENT_DIR}")

    # Verificar se há checkpoint para continuar treinamento
    checkpoint_data = load_checkpoint(EXPERIMENT_DIR)

    # Definir o dispositivo com base nas configurações
    device = torch.device("cuda" if torch.cuda.is_available() and not USE_CPU else "cpu")
    print_status_update(f"Usando dispositivo: {device}")

    if torch.cuda.is_available():
        print_status_update(f"GPU disponível: {torch.cuda.get_device_name(0)}")
        print_status_update(f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print_status_update(f"Memória GPU alocada: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Variáveis iniciais para o caso de não haver checkpoint
    start_time = time.time()
    start_epoch = 0
    best_perplexity = float('inf')

    if checkpoint_data is None:
        # Usando as variáveis globais de py
        # Definir seed para reprodutibilidade
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)

        logger.info(f"Train dataloader: {len(train_dataloader)} batches")
        logger.info(f"Eval dataloader: {len(eval_dataloader)} batches")

        from peft import LoraConfig, get_peft_model
        from transformers import BitsAndBytesConfig, AutoConfig, get_scheduler
        from torch.optim import AdamW
        from torch.cuda.amp import GradScaler

        # Configuração de quantização e precisão (usando bf16 do YAML)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16  # ensure compute dtype matches model input dtype
        )

        # Carregar configuração do modelo
        model_name = MODEL_NAME if USE_LOCAL_MODEL else MODEL_NAME_HF
        print_status_update(f"Carregando configuração do modelo: {model_name}")
        model_config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )

        # Selecionar o tipo de modelo baseado no espaço de busca
        print_status_update(f"Inicializando modelo LLaMA com espaço de busca: {SEARCH_SPACE}")
        model_map = {
            "small": SuperNetLlamaForCausalLMSMALL,
            "medium": SuperNetLlamaForCausalLMMEDIUM,
            "layer": SuperNetLlamaForCausalLMLAYER,
            "large": SuperNetLlamaForCausalLMLARGE
        }

        if SEARCH_SPACE not in model_map:
            raise ValueError(f"Espaço de busca inválido: {SEARCH_SPACE}. Opções: {list(model_map.keys())}")

        model_cls = model_map[SEARCH_SPACE]

        # Instanciar o modelo
        print_status_update("Carregando modelo pré-treinado. Isso pode levar alguns minutos...")
        model = model_cls.from_pretrained(
            model_name,
            config=model_config,
            cache_dir=CACHE_DIR,
            quantization_config=bnb_config  # Aplicando configuração de quantização aqui
        )
        print_status_update("Modelo base carregado com sucesso!")

        # Mover modelo para o dispositivo
        print_status_update(f"Movendo modelo para {device}...")
        model = model.to(device)
        print_status_update(f"Modelo movido para o dispositivo: {device}")

        # Configurar otimizador
        print_status_update("Configurando otimizador...")
        optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        # Configurar scheduler
        num_training_steps = int(NUM_EPOCHS * len(train_dataloader) / GRADIENT_ACCUMULATION)
        warmup_steps = int(WARMUP_RATIO * num_training_steps)

        print_status_update(f"Total de passos de treinamento: {num_training_steps}")
        print_status_update(f"Passos de warmup: {warmup_steps}")

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        # Usar os dados do checkpoint
        print_status_update("Usando checkpoint para continuar treinamento...")
        model = checkpoint_data["model"].to(device)
        optimizer = checkpoint_data["optimizer"]
        lr_scheduler = checkpoint_data["lr_scheduler"]
        scaler = checkpoint_data["scaler"]
        start_epoch = checkpoint_data["next_epoch"]
        best_perplexity = checkpoint_data["best_perplexity"]

        # Ajustar o tempo inicial para manter a contagem correta
        start_time = time.time() - checkpoint_data["total_runtime"]

        num_training_steps = int(NUM_EPOCHS * len(train_dataloader) / GRADIENT_ACCUMULATION)

        print_status_update(f"Retomando treinamento a partir da época {start_epoch+1}")
        print_status_update(f"Melhor perplexidade até agora: {best_perplexity:.4f}")

    # Selecionar estratégia de amostragem para NAS
    print_status_update(f"Usando estratégia de amostragem: {SAMPLING_STRATEGY}")

    # Obter o espaço de busca
    if SEARCH_SPACE == "small":
        search_space = SmallSearchSpace(model.config, seed=SEED)
    elif SEARCH_SPACE == "medium":
        search_space = MediumSearchSpace(model.config, seed=SEED)
    elif SEARCH_SPACE == "layer":
        search_space = LayerSearchSpace(model.config, seed=SEED)
    else:  # large
        search_space = FullSearchSpace(model.config, seed=SEED)

    from transformers.training_args import TrainingArguments

    # Criar objeto de training_args apenas para compatibilidade com o sampler
    training_args = TrainingArguments(
        output_dir=EXPERIMENT_DIR,
        seed=SEED
    )

    sampler = RandomSampler(search_space.config_space, seed=training_args.seed)

    # Selecionar estratégia de treinamento
    strategies = {
        "standard": StandardStrategy(
            sampler=sampler,
            loss_function=loss_function,
        ),
        "sandwich": SandwichStrategy(
            sampler=sampler,
            loss_function=loss_function,
        ),
        "random": RandomStrategy(
            sampler=sampler,
            loss_function=loss_function,
        ),
        "random_linear": RandomLinearStrategy(
            sampler=sampler,
            loss_function=loss_function,
            total_number_of_steps=num_training_steps,
        ),
    }

    if SAMPLING_STRATEGY not in strategies:
        raise ValueError(f"Estratégia de amostragem inválida: {SAMPLING_STRATEGY}. Opções: {list(strategies.keys())}")

    training_strategy = strategies[SAMPLING_STRATEGY]

    # Iniciar loop de treinamento
    print_status_update("Iniciando treinamento...")
    progress_bar = tqdm(range(num_training_steps), desc="Treinamento", leave=True)

    # Ajustar a barra de progresso para refletir o progresso já feito
    if start_epoch > 0:
        steps_done = start_epoch * len(train_dataloader) // GRADIENT_ACCUMULATION
        for _ in range(steps_done):
            progress_bar.update(1)

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()
        print_status_update(f"\n{'='*50}\nIniciando Época {epoch+1}/{NUM_EPOCHS}\n{'='*50}")
        model.train()
        train_loss = 0
        batch_times = []

        for batch_idx, batch in enumerate(train_dataloader):
            batch_start_time = time.time()

            # Mover batch para o dispositivo
            batch = {k: v.to(device) for k, v in batch.items()}

            # Debug: checar se há NaN/Inf nos inputs
            for k, v in batch.items():
                if not torch.isfinite(v).all():
                    print_status_update(f"[DEBUG] NaN/Inf detectado em batch['{k}'] no batch {batch_idx}")
            # Debug: checar weight sharing (embeddings)
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                emb = model.model.embed_tokens.weight
                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    print_status_update(f"[DEBUG] NaN/Inf detectado em embeddings no batch {batch_idx}")
            # Treinamento normal sem precisão mista
            loss = training_strategy(model, batch, batch["labels"]) / GRADIENT_ACCUMULATION
            # Debug: checar se a saída do modelo tem NaN/Inf
            try:
                y_hat = model(**batch)
                if hasattr(y_hat, 'logits') and not torch.isfinite(y_hat.logits).all():
                    print_status_update(f"[DEBUG] NaN/Inf detectado em logits do modelo no batch {batch_idx}")
            except Exception as e:
                print_status_update(f"[DEBUG] Erro ao checar logits: {e}")
            # Debug: checar se a loss é válida
            if not torch.isfinite(loss):
                print_status_update(f"[DEBUG] NaN/Inf detectado na loss no batch {batch_idx}")
                torch.save(batch, f"batch_nan_{batch_idx}.pt")
            if hasattr(model.model.embed_tokens.weight, 'grad') and model.model.embed_tokens.weight.grad is not None:
                if torch.isnan(model.model.embed_tokens.weight.grad).any() or torch.isinf(model.model.embed_tokens.weight.grad).any():
                    print_status_update(f"[DEBUG] NaN/Inf detectado nos gradientes dos embeddings no batch {batch_idx}")

            # Implementar gradient accumulation para simular batches maiores
            # Dividir a loss por gradient_accumulation para manter a escala
            do_optimizer_step = (batch_idx + 1) % GRADIENT_ACCUMULATION == 0

            if do_optimizer_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            current_loss = loss
            train_loss += current_loss

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            if do_optimizer_step:
                progress_bar.update(1)

            # Log mais frequente e detalhado
            if batch_idx % 10 == 0 or batch_idx == len(train_dataloader) - 1:
                lr = lr_scheduler.get_last_lr()[0] if do_optimizer_step else optimizer.param_groups[0]['lr']
                avg_time = sum(batch_times[-10:]) / min(len(batch_times), 10) if batch_times else 0
                est_remaining = avg_time * (len(train_dataloader) - batch_idx) / 60  # minutos

                print_status_update(
                    f"Época {epoch+1}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx+1}/{len(train_dataloader)} | "
                    f"Loss: {current_loss:.4f} | "
                    f"LR: {lr:.7f} | "
                    f"Tempo/batch: {batch_time:.2f}s | "
                    f"Restante ~{est_remaining:.1f}min"
                )

                # Mostrar informação de memória GPU se disponível
                if torch.cuda.is_available():
                    print_status_update(
                        f"GPU Memória: "
                        f"Alocada: {torch.cuda.memory_allocated(0)/1e9:.2f}GB | "
                        f"Reservada: {torch.cuda.memory_reserved(0)/1e9:.2f}GB | "
                        f"Max Alocada: {torch.cuda.max_memory_allocated(0)/1e9:.2f}GB"
                    )

        # Avaliação
        print_status_update(f"\nAvaliando modelo no fim da época {epoch+1}...")
        model.eval()
        eval_metric = calculate_perplexity(model, eval_dataloader, device)

        # Calcular médias
        avg_train_loss = train_loss / (len(train_dataloader))
        perplexity = eval_metric["perplexity"]

        # Log da época com mais detalhes
        epoch_runtime = time.time() - epoch_start_time
        total_runtime = time.time() - start_time

        print_status_update(f"\n{'*'*25} Resumo da Época {epoch+1} {'*'*25}")
        print_status_update(f"Loss média: {avg_train_loss:.4f}")
        print_status_update(f"Perplexidade: {perplexity:.4f}")
        print_status_update(f"Tempo da época: {epoch_runtime/60:.1f} minutos")
        print_status_update(f"Tempo total: {total_runtime/60:.1f} minutos")
        print_status_update(f"{'*'*65}\n")

        # Salvar checkpoint ao final de cada época
        checkpoint_dir = os.path.join(EXPERIMENT_DIR, f"checkpoint_epoca_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print_status_update(f"Salvando checkpoint em: {checkpoint_dir}")

        # Salvar modelo e tokenizer
        model.save_pretrained(checkpoint_dir)
        alpaca_wrapper.tokenizer.save_pretrained(checkpoint_dir)

        # Salvar estado do otimizador
        optimizer_state_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(optimizer.state_dict(), optimizer_state_path)

        # Salvar estado do scheduler
        scheduler_state_path = os.path.join(checkpoint_dir, "scheduler.pt")
        torch.save(lr_scheduler.state_dict(), scheduler_state_path)

        # Salvar scaler se estiver usando precisão mista
        if FP16_TRAINING and scaler is not None:
            scaler_state_path = os.path.join(checkpoint_dir, "scaler.pt")
            torch.save(scaler.state_dict(), scaler_state_path)

        # Salvar informações sobre o estado atual do treinamento
        training_state = {
            "epoch": epoch + 1,  # Próxima época a ser executada
            "batch_idx": 0,      # Começar do primeiro batch na próxima vez
            "best_perplexity": min(best_perplexity, perplexity),
            "total_runtime": time.time() - start_time,
            "rng_states": {
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }
        }
        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        torch.save(training_state, training_state_path)

        !cp -r /content/output /content/drive/MyDrive/

        print_status_update("Checkpoint completo salvo com sucesso!")

    # Avaliação final no conjunto de teste
    print_status_update("Realizando avaliação final no conjunto de teste...")
    model.eval()
    test_metric = calculate_perplexity(model, test_dataloader, device)
    test_perplexity = test_metric["perplexity"]
    print_status_update(f"Perplexidade final no teste: {test_perplexity:.4f}")

    # Salvar resultados finais
    results = {
        "modelo": MODEL_NAME if USE_LOCAL_MODEL else MODEL_NAME_HF,
        "espaço_busca": SEARCH_SPACE,
        "estratégia": SAMPLING_STRATEGY,
        "épocas": NUM_EPOCHS,
        "época_inicial": start_epoch,
        "perplexidade_validação": float(perplexity),
        "perplexidade_teste": float(test_perplexity),
        "tempo_total": time.time() - start_time,
        "parâmetros": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

    result_path = os.path.join(EXPERIMENT_DIR, "resultados_finais.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)

    print_status_update(f"Resultados finais do treinamento salvos em: {result_path}")

    print_status_update(f"\n{'='*30} TREINAMENTO CONCLUÍDO {'='*30}")
    print_status_update(f"Perplexidade de teste final: {test_perplexity:.4f}")
    print_status_update(f"Tempo total de treinamento: {(time.time() - start_time)/60:.2f} minutos")
    print_status_update(f"Todos os resultados foram salvos em: {EXPERIMENT_DIR}")

    # Salvar último modelo completo
    model_final_dir = os.path.join(EXPERIMENT_DIR, "modelo_final")
    model.save_pretrained(model_final_dir)
    alpaca_wrapper.tokenizer.save_pretrained(model_final_dir)
    print_status_update(f"Modelo final salvo em: {model_final_dir}")

    return model, results


if __name__ == "__main__":
    print_status_update("="*80)
    print_status_update("Iniciando script de treinamento NAS para LLaMA...")
    print_status_update("="*80)
    model, results = train_llama_nas()
    print_status_update("Treinamento finalizado com sucesso!")