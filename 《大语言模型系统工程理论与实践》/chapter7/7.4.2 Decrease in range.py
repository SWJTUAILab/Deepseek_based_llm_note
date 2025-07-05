import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def prune_model_by_magnitude(
        model,
        pruning_ratio=0.5,
        structured=False
):
    """
    基于权重幅度对模型进行剪枝

    参数:
    - model: PyTorch模型
    - pruning_ratio: 要剪枝的参数比例
    - structured: 是否进行结构化剪枝

    返回:
    - 剪枝后的模型
    """
    # 收集所有权重参数
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:  # 只考虑权重矩阵
            if structured:
                # 结构化剪枝：计算每个输出神经元的L2范数
                norm = torch.norm(param, p=2, dim=1)
                weights.append(norm.detach().cpu().numpy())
            else:
                # 非结构化剪枝：考虑所有权重
                weights.append(param.data.abs().detach().cpu().numpy().flatten())

    # 将所有权重展平并排序
    all_weights = np.concatenate([w.flatten() for w in weights])
    threshold = np.percentile(all_weights, pruning_ratio * 100)

    # 应用剪枝
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            if structured:
                # 结构化剪枝：移除整个输出神经元
                norm = torch.norm(param, p=2, dim=1)
                mask = (norm > threshold).float().unsqueeze(1).expand_as(param)
                param.data.mul_(mask)
            else:
                # 非结构化剪枝：移除单个权重
                mask = (param.data.abs() > threshold).float()
                param.data.mul_(mask)

    # 计算剪枝后的稀疏度
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()

    sparsity = zero_params / total_params
    print(f"剪枝后的稀疏度: {sparsity:.4f}")

    return model


def prune_llm_with_llm_pruner(
        model_name="The path where the model compression package is decompressed",
        output_dir="The path where you want to save",
        pruning_ratio=0.3
):
    """
    使用LLM-Pruner对大语言模型进行结构化剪枝

    参数:
    - model_name: 预训练模型名称或路径
    - output_dir: 剪枝后模型的保存路径
    - pruning_ratio: 要剪枝的参数比例
    """
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 第一步：识别重要的结构单元
    # 这里简化为基于幅度的结构化剪枝
    pruned_model = prune_model_by_magnitude(
        model,
        pruning_ratio=pruning_ratio,
        structured=True
    )

    # 第二步：微调剪枝后的模型（实际应用中需要）
    # 这里省略微调步骤，实际应用中应该进行微调

    # 保存剪枝后的模型
    pruned_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"剪枝模型已保存到 {output_dir}")

    return pruned_model, tokenizer


# 使用示例
if __name__ == "__main__":
    pruned_model, tokenizer = prune_llm_with_llm_pruner()

    # 测试剪枝后的模型
    input_text = "人工智能的未来发展趋势是"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        output_ids = pruned_model.generate(
            input_ids,
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)
