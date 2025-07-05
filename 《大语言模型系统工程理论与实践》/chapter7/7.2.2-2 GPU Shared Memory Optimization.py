import torch


def optimized_attention(query, key, value, mask=None):
    # 将输入移至GPU
    query, key, value = query.cuda(), key.cuda(), value.cuda()
    if mask is not None:
        mask = mask.cuda()

    # 计算注意力分数
    batch_size, num_heads, seq_len, head_dim = query.size()

    # 使用torch.cuda.amp进行混合精度计算
    with torch.cuda.amp.autocast():
        # 矩阵乘法计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (head_dim ** 0.5)

        # 应用掩码
        if mask is not None:
            attention_scores = attention_scores + mask

        # Softmax归一化
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # 计算输出
        context_layer = torch.matmul(attention_probs, value)

    return context_layer


# 函数调用
query = torch.randn([2, 8, 16, 32])
key = torch.randn([2, 8, 16, 32])
value = torch.randn([2, 8, 16, 32])
context_layer = optimized_attention(query, key, value)
print('This is result of attention:', context_layer)
