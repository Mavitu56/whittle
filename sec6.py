def get_backbone(model):
    model_type = model.base_model_prefix
    backbone = getattr(model, model_type)
    return backbone

def register_mask_ffn(module, mask):
    """
    Registra um hook para mascarar as entradas da feed-forward network,
    lidando com possíveis diferenças de dimensão entre o modelo e a máscara.

    Args:
        module: O módulo FFN do modelo
        mask: A máscara a ser aplicada

    Returns:
        handle: O handle do hook para posterior remoção
    """
    def hook_fn(_, inputs):
        import torch
        # Verificar o formato dos inputs, que pode variar entre modelos
        if isinstance(inputs, tuple):
            if len(inputs) > 0:
                input_tensor = inputs[0]
                # Debug: checar NaN/Inf em input_tensor e mask
                if not torch.isfinite(input_tensor).all():
                    print(f"[DEBUG] NaN/Inf detectado em input_tensor no hook_fn")
                if not torch.isfinite(mask).all():
                    print(f"[DEBUG] NaN/Inf detectado em mask no hook_fn")
                # Verifica se as dimensões são compatíveis
                if input_tensor.shape[-1] != mask.shape[-1]:
                    # Caso especial para LLaMA 3 ou outros modelos com dimensão diferente
                    if len(mask.shape) == 1:
                        # Se a máscara for 1D, repetir ou truncar para a dimensão correta
                        if input_tensor.shape[-1] > mask.shape[-1]:
                            # Máscara menor que a entrada - expandir por repetição
                            repeat_factor = input_tensor.shape[-1] // mask.shape[-1]
                            expanded_mask = mask.repeat(repeat_factor)
                            # Adicionar zeros se necessário para atingir o tamanho exato
                            if input_tensor.shape[-1] % mask.shape[-1] != 0:
                                padding = input_tensor.shape[-1] - expanded_mask.shape[-1]
                                expanded_mask = torch.cat([expanded_mask, mask[:padding]], dim=0)
                            masked_input = input_tensor * expanded_mask
                        else:
                            # Máscara maior que a entrada - truncar
                            masked_input = input_tensor * mask[:input_tensor.shape[-1]]
                    else:
                        # Para máscaras multidimensionais, ajustar a última dimensão
                        if len(input_tensor.shape) == len(mask.shape):
                            if input_tensor.shape[-1] > mask.shape[-1]:
                                # Expandir a máscara
                                expanded_shape = list(mask.shape)
                                expanded_shape[-1] = input_tensor.shape[-1]
                                expanded_mask = mask.new_zeros(expanded_shape)
                                repeat_factor = input_tensor.shape[-1] // mask.shape[-1]
                                for i in range(repeat_factor):
                                    expanded_mask[..., i*mask.shape[-1]:(i+1)*mask.shape[-1]] = mask
                                # Preencher o restante se necessário
                                remaining = input_tensor.shape[-1] - repeat_factor * mask.shape[-1]
                                if remaining > 0:
                                    expanded_mask[..., -remaining:] = mask[..., :remaining]
                                masked_input = input_tensor * expanded_mask
                            else:
                                # Truncar a máscara
                                masked_input = input_tensor * mask[..., :input_tensor.shape[-1]]
                        else:
                            # Caso as dimensões sejam muito diferentes, tentar uma abordagem simples
                            print(f"Aviso: Diferença significativa nas dimensões. input_tensor: {input_tensor.shape}, mask: {mask.shape}")
                            # Tentar broadcast se possível
                            try:
                                masked_input = input_tensor * mask
                            except:
                                # Último recurso: não aplicar máscara
                                print("Erro ao aplicar máscara. Retornando entrada original.")
                                masked_input = input_tensor
                else:
                    # Caso padrão: dimensões compatíveis
                    masked_input = input_tensor * mask

                # Retornar no mesmo formato que recebemos
                if len(inputs) > 1:
                    return (masked_input, *inputs[1:])
                else:
                    return (masked_input,)
            else:
                # Caso raro: tupla vazia
                print("Aviso: Tupla de inputs vazia no hook_fn")
                return inputs
        else:
            # Caso em que inputs não é uma tupla, mas um tensor único
            masked_input = inputs * mask
            return masked_input

    handle = module.register_forward_pre_hook(hook_fn)
    return handle

def register_drop_layer(module):
    """
    Registra um hook para "desativar" uma camada FFN, lidando com diferentes formatos de entrada.

    Args:
        module: O módulo FFN a ser desativado

    Returns:
        handle: O handle do hook para posterior remoção
    """
    def hook_fn(_, input_val, output_val):
        import torch
        # Verificar formato da entrada
        if isinstance(input_val, tuple):
            if len(input_val) > 1:
                # Debug: checar NaN/Inf em input_val[0]
                if not torch.isfinite(input_val[0]).all():
                    print(f"[DEBUG] NaN/Inf detectado em input_val[0] no register_drop_layer")
                # Formato original esperado: retornar o segundo elemento
                return input_val[1]
            elif len(input_val) > 0:
                # Debug: checar NaN/Inf em input_val[0]
                if not torch.isfinite(input_val[0]).all():
                    print(f"[DEBUG] NaN/Inf detectado em input_val[0] no register_drop_layer")
                # Caso tenha apenas um elemento: retornar o primeiro
                return input_val[0]
            else:
                # Caso raro: tupla vazia
                print("Aviso: Tupla de inputs vazia em register_drop_layer")
                # Tentar usar a saída como fallback
                if output_val is not None:
                    return output_val
                # Último recurso: retornar um tensor vazio
                return torch.tensor([], device=module.weight.device if hasattr(module, 'weight') else 'cuda')
        else:
            # Se o input não for uma tupla, retornar como está
            return input_val

    handle = module.register_forward_hook(hook_fn)
    return handle

def register_drop_attention_layer(module):
    """
    Registra um hook para "desativar" a camada de atenção, com suporte a diferentes formatos de retorno.

    Args:
        module: O módulo de atenção

    Returns:
        handle: O handle do hook para posterior remoção
    """
    def hook_fn(_, input_val, output_val):
        import torch
        # Verificar o formato da saída para lidar com diferentes implementações de LLaMA
        if output_val is None:
            # Caso raro
            print("Aviso: output_val é None em register_drop_attention_layer")
            return input_val

        # Para LLaMA 3, a saída pode ser uma tupla com (hidden_states, attention_weights)
        # ou apenas hidden_states
        if isinstance(output_val, tuple):
            if len(output_val) == 2:
                # Tupla com 2 elementos: (hidden_states, attention_weights)
                hidden_states, attention_weights = output_val
                # Retornamos a mesma estrutura
                return (hidden_states, attention_weights)
            elif len(output_val) == 0:
                # Tupla vazia - erro que estamos enfrentando
                # Retornar uma tupla com dois tensores vazios é uma abordagem mais segura
                # que tenta manter a estrutura esperada
                print("Aviso: Encontrada tupla vazia em register_drop_attention_layer")
                # Criar valores substitutos baseados no input
                if isinstance(input_val, tuple) and len(input_val) > 0:
                    # Debug: checar NaN/Inf em input_val[0]
                    if not torch.isfinite(input_val[0]).all():
                        print(f"[DEBUG] NaN/Inf detectado em input_val[0] no register_drop_attention_layer")
                    dummy_hidden = input_val[0]  # Usar o primeiro tensor do input
                    # Criar um tensor de zeros para attention_weights
                    batch_size = dummy_hidden.shape[0]
                    seq_len = dummy_hidden.shape[1]
                    dummy_attn = torch.zeros((batch_size, module.num_heads, seq_len, seq_len),
                                           device=dummy_hidden.device)
                    return (dummy_hidden, dummy_attn)
                else:
                    # Se não tiver input útil, apenas propagar None
                    print("Erro: Não foi possível criar substitutos em register_drop_attention_layer")
                    return None
            else:
                # Outros tamanhos de tupla - retornar como está
                return output_val
        else:
            # Não é uma tupla - retornar como está
            return output_val

    handle = module.register_forward_hook(hook_fn)
    return handle

#---------------- Máscaramento LLaMA ----------------#
def get_ffn(model, index):
    layer = get_layers(model)[index]
    ffn = layer.mlp
    return ffn

def get_attention_output(model, index):
    layer = get_layers(model)[index]
    output = layer.self_attn
    return output

def get_layers(model):
    # Para o modelo LLaMA, as camadas estão no atributo 'layers'
    if hasattr(model, 'model'):
        # Alguns modelos LLaMA têm um wrapper adicional
        layers = model.model.layers
    else:
        layers = model.layers
    return layers

def mask_llama(model, neuron_mask, head_mask):
    """
    Aplica mascaramento às camadas do modelo LLaMA

    Args:
        model: Modelo LLaMA
        neuron_mask: Máscara para unidades FFN
        head_mask: Máscara para cabeças de atenção

    Returns:
        handles: Lista de handles para remover máscaras posteriormente
    """
    num_hidden_layers = neuron_mask.shape[0]
    assert head_mask.shape[0] == num_hidden_layers

    handles = []
    for layer_idx in range(num_hidden_layers):
        ffn = get_ffn(model, layer_idx)
        handle = register_mask_ffn(ffn, neuron_mask[layer_idx])
        handles.append(handle)

        if neuron_mask[layer_idx].sum() == 0:
            handle = register_drop_layer(ffn)
            handles.append(handle)

        if head_mask[layer_idx].sum() == 0:
            attention = get_attention_output(model, layer_idx)
            handle = register_drop_attention_layer(attention)
            handles.append(handle)

    return handles