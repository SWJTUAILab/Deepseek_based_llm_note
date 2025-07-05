def estimate_throughput(
        model_params=7e9,
        hidden_size=4096,
        num_layers=32,
        seq_length=2048,
        batch_size=1,
        precision="fp16",
        gpu_tflops=312,
        gpu_memory_bandwidth=1555,
        compute_efficiency=0.3,
        memory_efficiency=0.7
):
    # 估算延迟
    latency = estimate_inference_latency(
        model_params=model_params,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_length=seq_length,
        batch_size=batch_size,
        precision=precision,
        gpu_tflops=gpu_tflops,
        gpu_memory_bandwidth=gpu_memory_bandwidth,
        compute_efficiency=compute_efficiency,
        memory_efficiency=memory_efficiency
    )

    # 计算吞吐量（tokens per second）
    throughput = batch_size * seq_length / latency["total_latency"]

    return throughput


# 估算不同批大小下的吞吐量
batch_sizes = [1, 2, 4, 8, 16]
for batch_size in batch_sizes:
    throughput = estimate_throughput(batch_size=batch_size)
    print(f"批大小: {batch_size}, 吞吐量: {throughput:.2f} tokens/s")
