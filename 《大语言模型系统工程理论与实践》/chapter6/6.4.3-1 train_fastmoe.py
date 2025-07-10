import os

import torch
import torch.nn as nn

# 启用GPU训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from fmoe.layers import FMoE


class MoETransformerLayer(nn.Module):
    """
    使用FastMoE的Transformer层
    """
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        num_experts=8,
        top_k=2,
    ):
        super(MoETransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # 定义专家模块的辅助类
        class FeedForwardExpert(nn.Module):
            def __init__(self, d_model, d_ffn):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.GELU(),
                    nn.Linear(d_ffn, d_model)
                )
            
            def forward(self, x, *args, **kwargs):
                return self.net(x)
        
        # 使用FastMoE，通过lambda传递参数
        self.moe = FMoE(
            num_expert=num_experts,
            d_model=hidden_size,
            top_k=top_k,
            expert=lambda d_model: FeedForwardExpert(d_model, ffn_hidden_size)
        )
        
        # 丢弃层
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None):
        """
        Transformer层的前向传播。
        """
        # 自注意力
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.dropout(x)
        x = residual + x
        
        # MoE前馈网络
        residual = x
        x = self.norm2(x)

        # 保存原始形状并展平
        original_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)

        # FMoE前向传播
        moe_output = self.moe(x_flat)

        # 恢复原始形状
        x = moe_output.view(original_shape)
        

        x = self.dropout(x)
        x = residual + x

        return x


def create_random_training_data(batch_size=16, seq_len=64, hidden_size=512, num_batches=10, device='cpu'):
    """创建随机训练数据"""
    training_data = []
    
    for i in range(num_batches):
        # 创建随机输入数据 (seq_len, batch_size, hidden_size)
        inputs = torch.randn(seq_len, batch_size, hidden_size, device=device)
        # 创建随机标签
        labels = torch.randn(seq_len, batch_size, hidden_size, device=device)
        training_data.append((inputs, labels))
    
    return training_data


def main():
    # 设置分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # 使用一个未被占用的端口
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    # 初始化分布式进程组
    if torch.cuda.is_available():
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://'
        )

    # 设置设备为CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建FastMoE模型 - 使用更小的参数以节省GPU内存
    model = MoETransformerLayer(
        hidden_size=1024,        # 减小隐藏层大小
        ffn_hidden_size=1024,    # 减小前馈网络大小
        num_attention_heads=4,  # 减少注意力头数
        num_experts=128,
        top_k=2
    ).to(device)
    
    print("FastMoE模型创建成功！")
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练数据
    print("正在创建随机训练数据...")
    training_data = create_random_training_data(
        batch_size=4,           # 减小批次大小
        seq_len=16,             # 减小序列长度
        hidden_size=1024,        # 匹配模型的隐藏大小
        num_batches=3,          # 减少批次数量
        device=str(device)      # 转换为字符串
    )
    print(f"训练数据创建完成！共 {len(training_data)} 个批次")
    
    # 设置训练参数
    num_epochs = 2
    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\n开始训练，共 {num_epochs} 个epoch...")
    
    # 训练循环（
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (train_inputs, train_labels) in enumerate(training_data):
    
            train_inputs = train_inputs.to(device)
            train_labels = train_labels.to(device)
            
            # 前向传播
            outputs = model(train_inputs)
            
            # 计算损失
            loss = criterion(outputs, train_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 2 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / len(training_data)
        print(f"Epoch [{epoch+1}/{num_epochs}] 完成, 平均损失: {avg_loss:.6f}")
    
    print("\n训练完成！")
    
    # 测试模型
    print("开始测试模型...")
    model.eval()
    with torch.no_grad():
        test_inputs = torch.randn(8, 2, 1024).to(device)  # (seq_len, batch_size, hidden_size) - 小尺寸
        test_outputs = model(test_inputs)
        print(f"测试输入形状: {test_inputs.shape}")
        print(f"测试输出形状: {test_outputs.shape}")
        print(f"测试输入设备: {test_inputs.device}")
        print(f"测试输出设备: {test_outputs.device}")
    
    print("模型测试完成！")
    
    # 展示训练数据统计信息
    if len(training_data) > 0:
        sample_input, sample_label = training_data[0]
        print(f"\n训练数据信息:")
        print(f"- 批次数量: {len(training_data)}")
        print(f"- 输入形状: {sample_input.shape}")
        print(f"- 标签形状: {sample_label.shape}")
        print(f"- 输入数据范围: [{sample_input.min():.3f}, {sample_input.max():.3f}]")
        print(f"- 标签数据范围: [{sample_label.min():.3f}, {sample_label.max():.3f}]")
    
    print("\n程序执行完成！")


if __name__ == "__main__":
    main()
