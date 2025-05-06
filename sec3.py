# Configuração de Variáveis
import os
import sys
import gc
import random
import numpy as np
import torch
from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaForCausalLM,
)
import math
import warnings
from typing import Any
from syne_tune.config_space import choice, lograndint, randint
from syne_tune.config_space import Categorical, Domain
import logging
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, get_scheduler, DataCollatorWithPadding, default_data_collator, DataCollatorForSeq2Seq, BitsAndBytesConfig
import json
import os
import requests
from datasets import Dataset
from dataclasses import dataclass, field
import evaluate
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from tqdm.auto import tqdm
import time
from torch.optim import AdamW
import transformers
from whittle.sampling.random_sampler import RandomSampler
from whittle.training_strategies import (
    RandomLinearStrategy,
    RandomStrategy,
    SandwichStrategy,
    StandardStrategy,
)

# Debug: checar variáveis globais para NaN/Inf
import torch
for var_name, var in globals().items():
    if isinstance(var, float) and (torch.isnan(torch.tensor(var)) or torch.isinf(torch.tensor(var))):
        print(f"[DEBUG] NaN/Inf detectado em variável global '{var_name}'")