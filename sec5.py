# Configuração de Variáveis
import os
import sys
import random
import numpy as np
import torch

# Definição de seed para reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configurações do modelo
MODEL_NAME_HF = "meta-llama/Llama-3.2-1B"  # Nome do modelo na HuggingFace (para referência)
MODEL_NAME = "/content/drive/MyDrive/Llama3-1B"  # Caminho local para o modelo
USE_LOCAL_MODEL = True  # Flag para usar o modelo local

# Tarefa e configuração de dados
TASK_NAME = "alpaca"  # Opções: "alpaca", "glue", "imdb", "swag"
# Configuração para o dataset Alpaca
ALPACA_TASK_INFO = {
    "alpaca": {
        "metric": "perplexity",
        "mode": "min",
        "seq_length": 128,
        "keys": ("instruction", "output"),
    }
}
MAX_SEQ_LENGTH = 128  # Conforme YAML: max_seq_length
DATA_SUBSET_SIZE = None  # Define como None para usar o dataset completo com filtragem automática

# Configuração do espaço de busca e estratégia de pruning
SEARCH_SPACE = "small"         # Opções: "small", "medium", "layer", "large"
SAMPLING_STRATEGY = "standard"   # Opções: "standard", "sandwich", "random", "random_linear"

# Hiperparâmetros de treinamento
LEARNING_RATE = 0.0001  # Reduzido de 0.0002 para um valor mais conservador
BATCH_SIZE = 32          # Conforme YAML (train.micro_batch_size)
GRADIENT_ACCUMULATION = 128  # Para simular batches maiores de tamanho 8 (train.global_batch_size)
NUM_EPOCHS = 5  # Conforme YAML (train.epochs)
# Usar um valor fixo para warmup (10%) em vez de cálculo baseado no tamanho do dataset
WARMUP_RATIO = 0.05  # 10% dos passos de treinamento serão para warmup
WEIGHT_DECAY = 0.01  # Conforme YAML (optimizer.init_args.weight_decay)
MAX_GRAD_NORM = 0.01      # Clipping de gradiente com valor mais conservador (era 0.5)

# Configurações de hardware e otimização de memória
USE_CPU = False                # Force CPU mesmo se GPU estiver disponível (para debug)
USE_8BIT_QUANTIZATION = True   # Usar quantização de 8 bits para economizar memória
GRADIENT_CHECKPOINTING = True  # Economiza memória durante o treinamento
FP16_TRAINING = False           # Usar precisão mista para economizar memória
BF16_TRAINING = False          # Alternativa para precisão mista (melhor em algumas GPUs)

# Configurações de avaliação
EVAL_STEPS = 100          # Número de steps entre avaliações (conforme YAML: eval.interval)
SAVE_STEPS = 200          # Número de steps entre salvamentos do modelo (conforme YAML: train.save_interval)
EVAL_BATCH_SIZE = 16       # Tamanho do batch para avaliação

# Configurações de diretórios
BASE_DIR = "/content"  # Usar o diretório atual como base
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DRIVE_BACKUP_DIR = os.path.join(BASE_DIR, "backups")  # Diretório para backup

# Criar diretórios necessários
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)

# Obter o nome do modelo para logs e diretórios
MODEL_NAME_SHORT = MODEL_NAME_HF.split('/')[-1] if not USE_LOCAL_MODEL else os.path.basename(MODEL_NAME)

# Variáveis específicas para experimentos
EXPERIMENT_NAME = f"pruning_{MODEL_NAME_SHORT}_{TASK_NAME}_{SEARCH_SPACE}_{SAMPLING_STRATEGY}"
EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# Debug: checar variáveis de configuração para NaN/Inf
for var_name, var in list(globals().items()):
    if isinstance(var, float) and (torch.isnan(torch.tensor(var)) or torch.isinf(torch.tensor(var))):
        print(f"[DEBUG] NaN/Inf detectado em variável de configuração '{var_name}'")

print(f"Configuração concluída para experimento: {EXPERIMENT_NAME}")
print(f"Diretório de saída: {EXPERIMENT_DIR}")
print(f"Usando modelo {'local' if USE_LOCAL_MODEL else 'remoto'}: {MODEL_NAME}")
print(f"Usando GPU: {torch.cuda.is_available() and not USE_CPU}")
if torch.cuda.is_available() and not USE_CPU:
    print(f"Dispositivo: {torch.cuda.get_device_name(0)}")