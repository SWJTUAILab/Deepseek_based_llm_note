import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)
        
        q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        if mask is not None:
            # 将掩码扩展为 [batch_size, 1, seq_len, seq_len]，以适配多头注意力
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, self.d_model)
        output = self.W_o(output)
        return output, attn

class SelfAttention(nn.Module):
    """
    自注意力机制
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, n_heads)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, seq_len, seq_len]
            
        返回:
            output: 自注意力输出 [batch_size, seq_len, d_model]
            attn: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        # 在自注意力中，Q、K、V都是同一个输入
        return self.multihead_attn(x, x, x, mask)
    
# 示例参数
batch_size = 2
seq_len = 10  # 序列长度
d_model = 512  # 模型维度
n_heads = 8  # 注意力头数

# 创建随机输入
x = torch.randn(batch_size, seq_len, d_model)

# 创建掩码（可选）
# 在自注意力中，掩码通常用于屏蔽未来位置（如在解码器中）
mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1).bool()  # 上三角掩码

# 初始化自注意力机制
self_attn = SelfAttention(d_model=d_model, n_heads=n_heads)

# 前向传播
output, attn = self_attn(x, mask)

print(f"输出形状: {output.shape}")  # [batch_size, seq_len, d_model]
print(f"注意力权重形状: {attn.shape}")  # [batch_size, n_heads, seq_len, seq_len]
