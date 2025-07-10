import torch
import torch.nn as nn
import torch.nn.functional as F

class GShardRouter(nn.Module):
    """GShard的Top-2路由器"""
    def __init__(self, input_size, num_experts, capacity_factor=1.1):
        super(GShardRouter, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
        # 路由器权重
        self.router = nn.Linear(input_size, num_experts, bias=False)
        
        # 初始化路由器权重
        nn.init.zeros_(self.router.weight)
    
    def forward(self, x, is_training=True):
        batch_size, seq_len, _ = x.shape
        
        # 计算路由logits
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # 计算路由概率
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
        
        # 选择概率最高的两个专家（Top-2路由）
        top2_probs, top2_indices = torch.topk(router_probs, k=2, dim=-1)  # [batch_size, seq_len, 2]
        
        # 归一化Top-2概率
        top2_probs = top2_probs / top2_probs.sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 2]
        
        # 计算每个专家的容量
        # 容量 = 每批次token数 * 容量因子 * 2 / 专家数量
        tokens_per_batch = batch_size * seq_len
        capacity = int(tokens_per_batch * self.capacity_factor * 2 / self.num_experts)
        
        # 创建分发张量和组合张量
        dispatch_tensor = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        combine_tensor = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        
        # 计算每个专家分配的token数量
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(2):
                    expert_idx = top2_indices[i, j, k].item()
                    expert_counts[expert_idx] += 1
        
        # 为每个专家分配token，考虑容量限制
        for expert_idx in range(self.num_experts):
            # 收集分配给当前专家的所有(batch_idx, seq_idx, k_idx)
            indices = []
            probs = []
            for i in range(batch_size):
                for j in range(seq_len):
                    for k in range(2):
                        if top2_indices[i, j, k].item() == expert_idx:
                            indices.append((i, j, k))
                            probs.append(top2_probs[i, j, k].item())
            
            # 如果分配的token数量超过容量，选择概率最高的一部分
            if len(indices) > capacity:
                # 按概率排序
                sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
                indices = [indices[i] for i in sorted_indices[:capacity]]
                probs = [probs[i] for i in sorted_indices[:capacity]]
            
            # 更新分发张量和组合张量
            for idx, (i, j, k) in enumerate(indices):
                dispatch_tensor[i, j, expert_idx] = 1.0
                combine_tensor[i, j, expert_idx] = top2_probs[i, j, k]
        
        # 计算负载均衡损失
        # 计算每个专家的分配概率
        router_prob_per_expert = router_probs.sum(dim=[0, 1]) / (batch_size * seq_len)
        # 计算每个专家的实际使用率
        expert_usage = expert_counts / expert_counts.sum()
        # 负载均衡损失
        aux_loss = (router_prob_per_expert * expert_usage).sum() * self.num_experts
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss

# 使用示例
if __name__ == "__main__":
    # 创建一个小型的测试样例
    batch_size = 4
    seq_len = 16
    input_size = 512
    num_experts = 8
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 创建路由器
    router = GShardRouter(input_size, num_experts)
    
    # 前向传播
    dispatch, combine, probs, loss = router(x)
    
    print(f"输入形状: {x.shape}")
    print(f"分发张量形状: {dispatch.shape}")
    print(f"组合张量形状: {combine.shape}")
    print(f"路由概率形状: {probs.shape}")
    print(f"辅助损失: {loss.item():.4f}")
