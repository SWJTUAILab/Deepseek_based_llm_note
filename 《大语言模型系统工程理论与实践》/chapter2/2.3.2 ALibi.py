# 概念性代码
import torch
def get_alibi_slopes(num_heads):
      # 计算每个头的斜率m_h
      # 通常是 2 的幂次方递减，例如 1/2, 1/4, 1/8, ...
      # 或者更精确地，使得 2^-(k * log2(num_heads) / (num_heads-1)) for k in range(num_heads)
      # 这里使用一个简化的等比数列
      base = 2**(-8.0 / num_heads)
      slopes = torch.pow(base, torch.arange(num_heads).float())
      return slopes
def alibi_bias(q_len, k_len, num_heads):
      """
      生成ALiBi的位置偏置矩阵，（适用于记忆力分数）
      核心逻辑：1.计算查询-键的相对距离
               2.每个注意力头使用不同斜率，将距离转换为负偏置
               偏置矩阵形状需匹配注意力分数维度：(batch, heads, q_len, k_len)
      """
      slopes = get_alibi_slopes(num_heads).unsqueeze(1).unsqueeze(2) # (num_heads, 1，1)
      # 距离矩阵 |i-j|
      """
      生成相对距离矩阵：
      行索引：查询位置 i ∈ [0, q_len-1]
      列索引：键位置 j ∈ [0, k_len-1]
      距离矩阵元素为 |i-j|，形状为(q_len, k_len)
      """
      distance = torch.arange(k_len)[None, :] - torch.arange(q_len)[:, None] # (q_len, k_len)
      distance = torch.abs(distance)
      bias = slopes * distance.unsqueeze(0) # (num_heads, q_len, k_len)
      return bias.unsqueeze(0) # (1, num_heads, q_len, k_len)

# 示例使用
num_heads = 8
q_len, k_len = 50, 60
attention_scores = torch.randn(1, num_heads, q_len, k_len)
alibi = alibi_bias(q_len, k_len, num_heads)
attention_scores_with_alibi = attention_scores - alibi  # 通常斜率为负，所以是减去
