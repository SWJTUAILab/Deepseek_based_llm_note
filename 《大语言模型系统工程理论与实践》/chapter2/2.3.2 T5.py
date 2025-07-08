# 概念性代码，非完整T5实现
import torch
import torch.nn as nn
class T5RelativePositionBias(nn.Module):
    def __init__(self, num_buckets, num_heads, max_distance):
        super().__init__()
        #嵌入层，存储每个桶对应的偏置值（一个注意力头一个值）
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
    def _relative_position_bucket(self, relative_position, bidirectional=True):
        num_buckets = self.num_buckets
        max_exact = num_buckets // 2 #前半部分桶用于精确表示近距离
        is_small = relative_position < max_exact
        # ... (省略了详细的分桶逻辑，T5有特定的分桶策略)
        # 简化版：将相对位置映射到桶索引
        # 将相对位置转换到非负范围，并确保不超过最大范围
        relative_buckets = torch.clamp(relative_position + self.max_distance, 0, 2 * self.max_distance -1)
        # 实际T5分桶更复杂，会考虑对数距离等
        return relative_buckets % num_buckets # 简化示例
    def forward(self, q_len, k_len):
       # q_len: query 序列长度, k_len: key 序列长度；该函数用来计算相对位置偏置。

       #创建位置将矩阵
        context_position = torch.arange(q_len, dtype=torch.long)[:, None]
        memory_position = torch.arange(k_len, dtype=torch.long)[None, :]
        #计算相对位置矩阵: 每个query位置与所有key位置的相对距离
        relative_position = memory_position - context_position  # shape (q_len, k_len)
        #将相对位置映射到桶索引
        rp_bucket = self._relative_position_bucket(relative_position)
        # 查找每个桶对应的偏置值
        bias = self.relative_attention_bias(rp_bucket) # shape (q_len, k_len, num_heads)
        # 调整维度，使其与注意力分数匹配
        bias = bias.permute(2, 0, 1).unsqueeze(0) # shape (1, num_heads, q_len, k_len)
        return bias
