def estimate_flops(
        hidden_size=4096,
        num_layers=32,
        seq_length=2048,
        batch_size=1
):
    # 自注意力FLOPs（每层）
    attention_flops_per_layer = (
        # QKV投影：3个 [B×S×H] × [H×H] 矩阵乘法
            3 * 2 * batch_size * seq_length * hidden_size * hidden_size +
            # 注意力分数计算：[B, num_heads, S, head_dim] × [B, num_heads, head_dim, S]
            2 * batch_size * seq_length * seq_length * hidden_size +
            # 注意力加权和：[B, num_heads, S, S] × [B, num_heads, S, head_dim]
            2 * batch_size * seq_length * seq_length * hidden_size +
            # 输出投影：[B×S×H] × [H×H]
            2 * batch_size * seq_length * hidden_size * hidden_size
    )

    # 整个模型的自注意力FLOPs
    attention_flops = num_layers * attention_flops_per_layer

    # 修正后的前馈网络FLOPs（每层）
    ffn_flops_per_layer = (
        # 第一个线性层：[B×S×H] × [H×4H]
            2 * batch_size * seq_length * hidden_size * (4 * hidden_size) +
            # 第二个线性层：[B×S×4H] × [4H×H]
            2 * batch_size * seq_length * (4 * hidden_size) * hidden_size
    )

    # 整个模型的前馈网络FLOPs
    ffn_flops = num_layers * ffn_flops_per_layer

    # 层归一化FLOPs（简化估计，实际值较小）
    # 注意：层归一化FLOPs ≈ 5 * B * S * H 每层
    ln_flops = batch_size * seq_length * num_layers * hidden_size * 5

    # 总FLOPs（包含所有组件）
    total_flops = attention_flops + ffn_flops + ln_flops

    # 转换为TFLOPS
    total_tflops = total_flops / 1e12

    return {
        "total_flops": total_flops,
        "total_tflops": total_tflops,
        "attention_tflops": attention_flops / 1e12,
        "ffn_tflops": ffn_flops / 1e12,
        "ln_tflops": ln_flops / 1e12
    }


# 计算LLaMA-7B的算力需求
flops_req = estimate_flops()
print(f"总计算量: {flops_req['total_tflops']:.2f} TFLOPs")
print(f"自注意力机制: {flops_req['attention_tflops']:.2f} TFLOPs")
print(f"前馈网络: {flops_req['ffn_tflops']:.2f} TFLOPs")
print(f"层归一化: {flops_req['ln_tflops']:.2f} TFLOPs")

