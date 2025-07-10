import os

from fmoe.layers import FMoE

# 强制使用CPU的环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 隐藏所有CUDA设备
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

import deepspeed
import torch
import torch.nn as nn
from deepspeed.moe.layer import MoE

# 强制torch使用CPU
torch.cuda.is_available = lambda: False

# 设置环境变量以避免分布式问题
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'


class MoETransformerLayer(nn.Module):
    """
    集成了MoE的Transformer层。
    """

    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        num_experts=8,
        top_k=2,
        ep_size=1,
        use_residual=False,
    ):
        super(MoETransformerLayer, self).__init__()

        # 多头注意力
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads)

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # 定义专家模块
        expert = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Linear(ffn_hidden_size, hidden_size),
        )

        # 使用FastMoE
        # self.moe = FMoE(
        #     num_expert=num_experts,
        #     d_model=hidden_size,
        #     top_k=top_k,
        #     expert=expert
        # )

        # 使用DeepSpeed MoE

        self.moe = MoE(
            hidden_size=hidden_size,
            expert=expert,
            num_experts=num_experts,
            ep_size=ep_size,
            k=top_k,
            use_residual=use_residual,
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
        moe_output = self.moe(x)
        # DeepSpeed MoE可能返回元组 (output, aux_loss) 或只返回output
        if isinstance(moe_output, tuple):
            x = moe_output[0]  # 只取输出，忽略辅助损失
        else:
            x = moe_output
        x = self.dropout(x)
        x = residual + x

        return x


# 设置设备为CPU
device = torch.device('cpu')
print(f"使用设备: {device}")

# 创建MoE模型
model = MoETransformerLayer(
    hidden_size=512,          # 隐藏层大小
    ffn_hidden_size=2048,     # 前馈网络隐藏层大小
    num_attention_heads=8,    # 注意力头数
    num_experts=4,            # 专家数量
    top_k=2,                  # 每次选择的专家数量
    ep_size=1,                # 专家并行大小
    use_residual=False        # 是否使用残差连接
)

# 将模型移动到CPU
model = model.to(device)


print("MoE模型创建成功！")
print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

# 创建参数组（对于DeepSpeed MoE）
def create_moe_param_groups(model):
    from deepspeed.moe.utils import \
        split_params_into_different_moe_groups_for_optimizer
    parameters = {"params": [p for p in model.parameters()], "name": "parameters"}
    return split_params_into_different_moe_groups_for_optimizer(parameters)


parameters = create_moe_param_groups(model)

# DeepSpeed配置 - 针对CPU优化
ds_config = {
    "train_batch_size": 16,
    "fp16": {
        "enabled": False  # CPU不支持FP16
    },
    "bf16": {
        "enabled": False  # CPU一般不支持BF16
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "moe": {
        "enabled": True,
        "min_capacity": 4,
        "capacity_factor": 1.0
    }
}

# 初始化分布式环境（单进程）
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(
        backend='gloo',  # 使用gloo后端支持CPU
        init_method='env://',
        world_size=1,
        rank=0
    )

# 创建优化器参数
model_parameters = create_moe_param_groups(model)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model_parameters,
    config=ds_config,
    dist_init_required=False  # 
)



# 创建随机训练数据
def create_random_training_data(batch_size=32, seq_len=128, hidden_size=512, num_batches=50, device='cpu'):
    """
    创建随机训练数据
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        hidden_size: 隐藏层维度
        num_batches: 训练批次数量
        device: 设备类型
    
    Returns:
        训练数据列表，每个元素包含(输入, 标签)
    """
    training_data = []
    
    for i in range(num_batches):
        # 创建随机输入数据 (seq_len, batch_size, hidden_size)
        # 这是MultiheadAttention期望的输入格式
        inputs = torch.randn(seq_len, batch_size, hidden_size, device=device)
        
        # 创建随机标签，这里使用与输入相同的维度
        # 在实际应用中，这可能是下一个词的预测目标等
        labels = torch.randn(seq_len, batch_size, hidden_size, device=device)
        
        training_data.append((inputs, labels))
    
    return training_data

# 创建训练数据
# 确保模型和数据都在CPU上的函数
def ensure_cpu_device(model_engine):
    """确保模型在CPU设备上"""
    model_engine.module.cpu()
    for param in model_engine.module.parameters():
        param.data = param.data.cpu()
    return model_engine

model_engine = ensure_cpu_device(model_engine)

print("正在创建随机训练数据...")
training_data = create_random_training_data(
    batch_size=16,    # 批次大小
    seq_len=64,       # 序列长度
    hidden_size=512,  # 隐藏层维度（与模型一致）
    num_batches=20,   # 训练批次数量
    device='cpu'      # 使用CPU设备
)
print(f"训练数据创建完成！共 {len(training_data)} 个批次")

# 定义训练参数
num_epochs = 5
# learning_rate = 0.001
print_every = 1

# 定义损失函数
criterion = nn.MSELoss()  # 均方误差损失


# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"开始训练，共 {num_epochs} 个epoch...")
print(f"使用 DeepSpeed 进行训练")

# 训练循环
for epoch in range(num_epochs):
    # DeepSpeed训练
    model_engine.train()
    epoch_loss = 0.0
    
    for batch_idx, (train_inputs, train_labels) in enumerate(training_data):
        # 前向传播
        outputs = model_engine(train_inputs)
        
        # 计算损失
        loss = criterion(outputs, train_labels)
        
        # DeepSpeed反向传播
        model_engine.backward(loss)
        model_engine.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 5 == 0:
            print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")

    
    # 打印epoch总结
    avg_loss = epoch_loss / len(training_data)
    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] 完成, 平均损失: {avg_loss:.6f}")

print("训练完成！")

# 测试模型
print("\n开始测试模型...")

model_engine.eval()
with torch.no_grad():
    # 创建测试数据 - 确保在CPU上
    test_inputs = torch.randn(32, 8, 512, device=device)  # (seq_len, batch_size, hidden_size)
    test_outputs = model_engine(test_inputs)
    print(f"测试输入形状: {test_inputs.shape}")
    print(f"测试输出形状: {test_outputs.shape}")
    print(f"测试输入设备: {test_inputs.device}")
    print(f"测试输出设备: {test_outputs.device}")


print("模型测试完成！")

# 展示一些训练数据的统计信息
if len(training_data) > 0:
    sample_input, sample_label = training_data[0]
    print(f"\n训练数据信息:")
    print(f"- 批次数量: {len(training_data)}")
    print(f"- 输入形状: {sample_input.shape}")
    print(f"- 标签形状: {sample_label.shape}")
    print(f"- 输入数据范围: [{sample_input.min():.3f}, {sample_input.max():.3f}]")
    print(f"- 标签数据范围: [{sample_label.min():.3f}, {sample_label.max():.3f}]")

print("\n程序执行完成！")
