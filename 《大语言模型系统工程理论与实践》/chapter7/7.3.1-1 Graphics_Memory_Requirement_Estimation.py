def estimate_memory_requirements(
        model_params=7e9,  # 7B参数
        hidden_size=4096,
        num_layers=32,
        batch_size=1,
        seq_length=2048,
        precision="fp16"  # 可选: "fp32", "fp16", "int8", "int4"
):
    # 精度对应的字节数
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5  # 实际实现中通常为1字节存储2个int4值
    }

    bytes_per_element = precision_bytes[precision]

    # 计算参数显存
    params_memory = model_params * bytes_per_element

    # 计算KV缓存显存
    # 2表示K和V两个缓存
    kv_cache_memory = 2 * batch_size * seq_length * hidden_size * num_layers * bytes_per_element

    # 计算中间激活值显存（简化估计）
    activation_memory = 2 * batch_size * seq_length * hidden_size * bytes_per_element

    # 总显存需求
    total_memory = params_memory + kv_cache_memory + activation_memory

    # 转换为GB
    total_memory_gb = total_memory / (1024 ** 3)
    params_memory_gb = params_memory / (1024 ** 3)
    kv_cache_memory_gb = kv_cache_memory / (1024 ** 3)
    activation_memory_gb = activation_memory / (1024 ** 3)

    return {
        "total_memory_gb": total_memory_gb,
        "params_memory_gb": params_memory_gb,
        "kv_cache_memory_gb": kv_cache_memory_gb,
        "activation_memory_gb": activation_memory_gb
    }


# 计算不同精度下的显存需求
for precision in ["fp32", "fp16", "int8", "int4"]:
    memory_req = estimate_memory_requirements(precision=precision)
    print(f"精度: {precision}")
    print(f"  总显存需求: {memory_req['total_memory_gb']:.2f} GB")
    print(f"  参数显存: {memory_req['params_memory_gb']:.2f} GB")
    print(f"  KV缓存显存: {memory_req['kv_cache_memory_gb']:.2f} GB")
    print(f"  激活值显存: {memory_req['activation_memory_gb']:.2f} GB")
    print()
