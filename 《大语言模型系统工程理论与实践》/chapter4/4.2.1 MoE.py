import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertLayer(nn.Module):
    """单个专家网络，实现为简单的前馈网络"""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(ExpertLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MoELayer(nn.Module):
    """混合专家模型层 - 改进版本"""
    def __init__(self, input_size, output_size, num_experts, hidden_size, k=1, capacity_factor=1.0, dropout=0.1):
        super(MoELayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            ExpertLayer(input_size, hidden_size, output_size, dropout)
            for _ in range(num_experts)
        ])
        
        # 路由器
        self.router = nn.Linear(input_size, num_experts, bias=False)
        
        # 初始化路由器权重
        nn.init.zeros_(self.router.weight)
    
    def forward(self, x, is_training=True):
        batch_size, seq_len, _ = x.shape
        original_shape = x.shape
        
        # 重塑输入为 [batch_size * seq_len, input_size]
        x_flat = x.view(-1, self.input_size)
        
        # 计算路由概率
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        
        # 在训练时添加噪声以提高稳定性
        if is_training:
            router_logits += torch.randn_like(router_logits) * 0.1
        
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size * seq_len, num_experts]
        
        # 选择Top-k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.k, dim=-1)
        
        # 归一化Top-k概率
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 更高效的专家计算方法
        for expert_idx in range(self.num_experts):
            # 创建mask来标识分配给当前专家的tokens
            expert_mask = (top_k_indices == expert_idx)
            
            if expert_mask.any():
                # 找到所有分配给当前专家的位置
                expert_positions = expert_mask.nonzero(as_tuple=False)
                
                if expert_positions.size(0) > 0:
                    # 提取输入tokens
                    token_indices = expert_positions[:, 0]
                    k_indices = expert_positions[:, 1]
                    
                    expert_inputs = x_flat[token_indices]
                    expert_weights = top_k_probs[token_indices, k_indices]
                    
                    # 计算专家输出
                    expert_output = self.experts[expert_idx](expert_inputs)
                    
                    # 加权输出并累积到最终结果
                    weighted_output = expert_output * expert_weights.unsqueeze(-1)
                    output.index_add_(0, token_indices, weighted_output)
        
        # 重塑输出回原始形状
        output = output.view(original_shape)
        
        # 计算标准的负载均衡损失
        # P(expert) = 平均路由概率, f(expert) = 分配给专家的token比例
        mean_router_prob_per_expert = router_probs.mean(dim=0)  # [num_experts]
        
        # 计算每个专家实际处理的token比例
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx)
            tokens_per_expert[expert_idx] = expert_mask.sum().float() / (x_flat.size(0) * self.k)
        
        # 负载均衡损失：最小化 sum(P(expert) * f(expert)) * num_experts
        load_balancing_loss = torch.sum(mean_router_prob_per_expert * tokens_per_expert) * self.num_experts
        
        return output, load_balancing_loss

    def get_expert_utilization(self, x):
        """获取专家利用率统计信息"""
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.input_size)
        
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.k, dim=-1)
        
        # 统计每个专家的使用次数
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx)
            expert_counts[expert_idx] = expert_mask.sum().float()
        
        return {
            'expert_counts': expert_counts,
            'expert_utilization': expert_counts / expert_counts.sum(),
            'max_utilization': expert_counts.max() / expert_counts.sum(),
            'min_utilization': expert_counts.min() / expert_counts.sum()
        }

# 使用示例和测试
if __name__ == "__main__":
    # 创建一个小型的测试样例
    batch_size = 4
    seq_len = 16
    input_size = 512
    output_size = 512
    hidden_size = 1024
    num_experts = 8
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 创建MoE层
    moe_layer = MoELayer(input_size, output_size, num_experts, hidden_size, k=2)
    
    # 前向传播
    output, loss = moe_layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"负载均衡损失: {loss.item():.4f}")
    
    # 获取专家利用率统计
    stats = moe_layer.get_expert_utilization(x)
    print(f"专家利用率分布: {stats['expert_utilization']}")
    print(f"最大利用率: {stats['max_utilization']:.4f}")
    print(f"最小利用率: {stats['min_utilization']:.4f}")
    
    # 测试梯度计算
    print("\n测试梯度计算...")
    loss_total = loss + output.sum()  # 简单的损失函数
    loss_total.backward()
    
    # 检查是否所有专家都有梯度
    for i, expert in enumerate(moe_layer.experts):
        if expert.fc1.weight.grad is not None:
            print(f"专家 {i} 有梯度: 平均梯度大小 = {expert.fc1.weight.grad.abs().mean().item():.6f}")
        else:
            print(f"专家 {i} 没有梯度")
