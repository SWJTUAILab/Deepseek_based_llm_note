import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttention(nn.Module):
    """
    FlashAttention的简化实现
    
    参数:
        softmax_scale: softmax缩放因子，默认为1/sqrt(d_k)
        block_size: 分块大小
    """
    def __init__(self, softmax_scale=None, block_size=256):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.block_size = block_size
        
    def forward(self, q, k, v, mask=None):
        """
        前向传播
        
        参数:
            q: 查询向量 [batch_size, len_q, d_k]
            k: 键向量 [batch_size, len_k, d_k]
            v: 值向量 [batch_size, len_v, d_v]，其中len_k == len_v
            mask: 掩码 [batch_size, len_q, len_k]
            
        返回:
            output: 注意力输出 [batch_size, len_q, d_v]
        """
        batch_size, len_q, d_k = q.size()
        len_k, d_v = k.size(1), v.size(2)
        
        # 设置softmax缩放因子
        if self.softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(d_k)
        
        # 初始化输出和辅助变量
        output = torch.zeros(batch_size, len_q, d_v, device=q.device)
        
        # 计算分块数量
        num_blocks_q = (len_q + self.block_size - 1) // self.block_size
        num_blocks_k = (len_k + self.block_size - 1) // self.block_size
        
        # 初始化每个查询位置的最大值和缩放因子
        m = torch.ones(batch_size, len_q, 1, device=q.device) * float('-inf')
        l = torch.zeros(batch_size, len_q, 1, device=q.device)
        
        # 分块计算注意力
        for i in range(num_blocks_q):
            # 确定当前查询块的范围
            q_start = i * self.block_size
            q_end = min(len_q, (i + 1) * self.block_size)
            q_block = q[:, q_start:q_end, :]
            
            for j in range(num_blocks_k):
                # 确定当前键值块的范围
                k_start = j * self.block_size
                k_end = min(len_k, (j + 1) * self.block_size)
                k_block = k[:, k_start:k_end, :]
                v_block = v[:, k_start:k_end, :]
                
                # 计算当前块的注意力分数
                s_block = torch.bmm(q_block, k_block.transpose(1, 2)) * self.softmax_scale
                
                # 应用掩码（如果提供）
                if mask is not None:
                    mask_block = mask[:, q_start:q_end, k_start:k_end]
                    s_block = s_block.masked_fill(mask_block, float('-inf'))
                
                # 更新最大值
                m_block = m[:, q_start:q_end, :]
                m_new = torch.max(torch.cat([m_block, s_block.max(dim=2, keepdim=True)[0]], dim=2), dim=2, keepdim=True)[0]
                
                # 计算缩放因子
                exp_block = torch.exp(s_block - m_new)
                l_block = l[:, q_start:q_end, :]
                l_new = l_block * torch.exp(m_block - m_new) + exp_block.sum(dim=2, keepdim=True)
                
                # 更新输出
                output_block = output[:, q_start:q_end, :]
                output_new = (output_block * torch.exp(m_block - m_new) * l_block / l_new) + torch.bmm(exp_block / l_new, v_block)
                
                # 更新状态
                output[:, q_start:q_end, :] = output_new
                m[:, q_start:q_end, :] = m_new
                l[:, q_start:q_end, :] = l_new
        
        return output

# 示例参数
batch_size = 2
len_q = 1024  # 查询序列长度
len_k = 1024  # 键/值序列长度
d_k = 64  # 键向量维度
d_v = 64  # 值向量维度
block_size = 256  # 分块大小

# 创建随机输入
q = torch.randn(batch_size, len_q, d_k)
k = torch.randn(batch_size, len_k, d_k)
v = torch.randn(batch_size, len_k, d_v)

# 创建掩码（可选）
mask = torch.zeros(batch_size, len_q, len_k).bool()  # False表示不掩码

# 初始化FlashAttention
flash_attn = FlashAttention(block_size=block_size)

# 前向传播
output = flash_attn(q, k, v, mask)

print(f"输出形状: {output.shape}")  # [batch_size, len_q, d_v]