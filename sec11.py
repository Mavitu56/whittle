# Configurar logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Definindo os argumentos para inicializar os data wrappers
@dataclass
class DataArguments:
    task_name: str = field(default="alpaca")
    max_seq_length: int = field(default=512)
    pad_to_max_length: bool = field(default=True)
    dataset_seed: int = field(default=42)
    train_file: str = field(default=None)
    validation_file: str = field(default=None)
    test_file: str = field(default=None)
    max_train_samples: int = field(default=None)
    max_eval_samples: int = field(default=None)
    max_test_samples: int = field(default=None)
    preprocessing_num_workers: int = field(default=None)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3-8B-Instruct")
    cache_dir: str = field(default="cache")
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

logger.info(f"Preparando dados para tarefa: {TASK_NAME}")

# Se a tarefa for Alpaca, usar nossa classe de data wrapper implementada
if TASK_NAME == "alpaca":
    logger.info("Inicializando o Alpaca data wrapper")

# Criar os argumentos necessários
data_args = DataArguments(
    task_name=TASK_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    pad_to_max_length=True,
    dataset_seed=SEED,
    # Se estamos usando modelo remoto, vamos tentar baixar o dataset automaticamente
    train_file=os.path.join(BASE_DIR, "data", "alpaca", "alpaca_data.json") if not USE_LOCAL_MODEL else None,
)

model_args = ModelArguments(
    model_name_or_path=MODEL_NAME_HF if not USE_LOCAL_MODEL else MODEL_NAME,
    cache_dir=CACHE_DIR,
    use_fast_tokenizer=True,
)

# Criar os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    seed=SEED,
)

# Instanciar o data wrapper
alpaca_wrapper = Alpaca(
training_args=training_args,
model_args=model_args,
data_args=data_args,
)

# Obter os dataloaders
train_dataloader, eval_dataloader, test_dataloader = alpaca_wrapper.get_data_loaders()

# Acessar os datasets
train_dataset = alpaca_wrapper.train_data
eval_dataset = alpaca_wrapper.valid_data
test_dataset = alpaca_wrapper.test_data
data_collator = alpaca_wrapper.get_data_collator()

logger.info(f"Dataset de treino: {len(train_dataset)} exemplos")
logger.info(f"Dataset de validação: {len(eval_dataset)} exemplos")
logger.info(f"Dataset de teste: {len(test_dataset)} exemplos")

# Verificar formato dos dados para debugging
try:
    batch = next(iter(eval_dataloader))
    logger.info(f"Chaves do batch: {batch.keys()}")
    logger.info(f"Exemplo de labels: {batch['labels'][:2] if 'labels' in batch else 'Sem labels no batch'}")
    # Debug: checar NaN/Inf
    import torch
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            logger.warning(f"[DEBUG] NaN/Inf detectado em eval_dataloader['{k}']")
except Exception as e:
    logger.warning(f"Erro ao verificar formato do batch: {str(e)}")

# Configurar métricas para avaliação
logger.info("Configurando perplexidade como métrica para tarefa de geração")
import evaluate
metric = evaluate.load("perplexity")
metric_name = "perplexity"

logger.info("Preparação de dados concluída. Pronto para treinamento.")