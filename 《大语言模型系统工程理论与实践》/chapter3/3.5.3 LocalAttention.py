import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LocalAttention(nn.Module):
    """
    局部注意力机制
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        window_size: 局部窗口大小
    """
    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.scale = np.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, seq_len, seq_len]
            
        返回:
            output: 局部注意力输出 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. 线性投影
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_v]
        
        # 2. 创建局部注意力掩码
        local_mask = torch.ones(seq_len, seq_len, device=x.device).bool()
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            local_mask[i, start:end] = False
        
        # 3. 结合用户提供的掩码和局部掩码
        if mask is not None:
            combined_mask = mask | local_mask.unsqueeze(0)
        else:
            combined_mask = local_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 4. 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, n_heads, seq_len, seq_len]
        
        # 5. 应用掩码
        combined_mask = combined_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # [batch_size, n_heads, seq_len, seq_len]
        attn_scores = attn_scores.masked_fill(combined_mask, -np.inf)
        
        # 6. 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, n_heads, seq_len, seq_len]
        
        # 7. 计算输出
        output = torch.matmul(attn_weights, v)  # [batch_size, n_heads, seq_len, d_v]
        
        # 8. 重塑并投影回原始维度
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [batch_size, seq_len, n_heads * d_v]
        output = self.W_o(output)  # [batch_size, seq_len, d_model]
        
        return output

# 示例参数
batch_size = 2
seq_len = 100  # 序列长度
d_model = 512  # 模型维度
n_heads = 8  # 注意力头数
window_size = 16  # 局部窗口大小

# 创建随机输入
x = torch.randn(batch_size, seq_len, d_model)

# 创建掩码（可选）
mask = torch.zeros(batch_size, seq_len, seq_len).bool()  # False表示不掩码

# 初始化局部注意力机制
local_attn = LocalAttention(d_model=d_model, n_heads=n_heads, window_size=window_size)

# 前向传播
output = local_attn(x, mask)

print(f"输出形状: {output.shape}")  # 预期输出形状: [batch_size, seq_len, d_model]