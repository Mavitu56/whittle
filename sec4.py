# Verificar versão do PyTorch e disponibilidade de GPU
print(f"PyTorch versão: {torch.__version__}")
print(f"Transformers versão: {transformers.__version__}")
print(f"GPU disponível: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Mostrar informações da GPU
    device = torch.device("cuda")
    print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Memória GPU disponível: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

    # Debug: checar memória GPU para NaN/Inf
    mem_info = torch.cuda.mem_get_info()
    if any([not torch.isfinite(torch.tensor(m)) for m in mem_info]):
        print("[DEBUG] NaN/Inf detectado em memória GPU")

    # Configurações para economizar memória
    torch.cuda.empty_cache()
    gc.collect()

    # Configurar para uso eficiente de memória
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Evitar warnings de paralelismo
    torch.backends.cudnn.benchmark = True  # Otimizar operações de convolução

    print("Configurações de memória otimizadas para GPU ativadas")
else:
    device = torch.device("cpu")
    print("Usando CPU para processamento")

print(f"Dispositivo ativo: {device}")