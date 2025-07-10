def estimate_inference_latency(
        model_params=7e9,  # 7B参数
        hidden_size=4096,
        num_layers=16,
        seq_length=2048,
        batch_size=1,
        precision="fp16",
        gpu_tflops=312,  # A100 FP16理论峰值
        gpu_memory_bandwidth=1555,  # A100 HBM2内存带宽(GB/s)
        compute_efficiency=0.9,  # 计算效率
        memory_efficiency=0.1  # 内存访问效率
):
    # 计算FLOPs
    flops_estimate = estimate_flops(
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_length=seq_length,
        batch_size=batch_size
    )
    total_flops = flops_estimate["total_flops"]

    # 计算内存访问量
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5
    }
    bytes_per_element = precision_bytes[precision]

    # 模型参数读取
    params_bytes = model_params * bytes_per_element

    # KV缓存访问
    kv_cache_bytes = 2 * batch_size * seq_length * hidden_size * num_layers * bytes_per_element

    # 中间激活值访问（简化估计）
    activation_bytes = 2 * batch_size * seq_length * hidden_size * bytes_per_element

    # 总内存访问量
    total_memory_bytes = params_bytes + kv_cache_bytes + activation_bytes

    # 计算延迟估算
    compute_latency = total_flops / (gpu_tflops * 1e12 * compute_efficiency)

    # 内存访问延迟估算
    memory_latency = total_memory_bytes / (gpu_memory_bandwidth * 1e9 * memory_efficiency)

    # 总延迟（取较大值）
    total_latency = max(compute_latency, memory_latency)

    return {
        "total_latency": total_latency,
        "compute_latency": compute_latency,
        "memory_latency": memory_latency,
        "compute_bound": compute_latency > memory_latency
    }


# 估算LLaMA-7B在不同精度下的推理延迟
for precision in ["fp32", "fp16", "int8", "int4"]:
    latency = estimate_inference_latency(precision=precision)
    print(f"精度: {precision}")
    print(f"  总延迟: {latency['total_latency'] * 1000:.2f} ms")
    print(f"  计算延迟: {latency['compute_latency'] * 1000:.2f} ms")
    print(f"  内存延迟: {latency['memory_latency'] * 1000:.2f} ms")
    print(f"  瓶颈类型: {'计算受限' if latency['compute_bound'] else '内存受限'}")
    print()
