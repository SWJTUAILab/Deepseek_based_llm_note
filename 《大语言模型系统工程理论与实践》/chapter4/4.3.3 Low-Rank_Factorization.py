import torch
import torch.nn as nn
import types
import sys

# 尝试导入自定义模型
try:
    from model import is_pytorch_version_compatible, SimpleModel
    use_transformers = False
    
    # 如果PyTorch版本满足要求，尝试导入transformers
    if is_pytorch_version_compatible():
        try:
            from transformers import GPT2LMHeadModel, GPT2Config
            use_transformers = True
            print("使用transformers库的预训练模型")
        except:
            print("无法导入transformers库，将使用自定义模型")
    else:
        print("PyTorch版本不满足要求，将使用自定义模型")
except:
    # 如果无法导入自定义模型，直接尝试导入transformers
    try:
        from transformers import GPT2LMHeadModel, GPT2Config
        use_transformers = True
        print("使用transformers库的预训练模型")
    except:
        print("无法导入transformers库或自定义模型，程序无法继续")
        sys.exit(1)

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 低秩矩阵A和B
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        
        # 初始化
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)
    
    def forward(self, x):
        # 计算低秩更新：x @ (A @ B) * scaling
        return (x @ self.A @ self.B) * self.scaling

# 将LoRA应用到模型
def add_lora_to_gpt2(model, rank=4, alpha=1.0):
    # 遍历模型的所有线性层
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 获取原始线性层的输入和输出维度
            in_features = module.in_features
            out_features = module.out_features
            
            # 创建LoRA层
            lora_layer = LoRALayer(in_features, out_features, rank, alpha)
            
            # 保存原始前向传播函数
            original_forward = module.forward
            
            # 定义新的前向传播函数，添加LoRA更新
            def new_forward(self, x):
                original_output = original_forward(x)
                lora_output = lora_layer(x)
                return original_output + lora_output
            
            # 替换前向传播函数
            module.forward = types.MethodType(new_forward, module)
            
            # 将LoRA层添加为模块的属性
            setattr(module, 'lora', lora_layer)
    
    # 冻结原始模型参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 只训练LoRA参数
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            for param in module.lora.parameters():
                param.requires_grad = True
    
    return model

print("加载模型...")
if use_transformers:
    try:
        # 使用transformers的预训练模型
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        print("成功加载GPT2预训练模型")
    except Exception as e:
        print(f"无法加载预训练模型: {e}")
        print("切换到自定义模型")
        use_transformers = False
        model = SimpleModel(vocab_size=50257, hidden_size=768, num_layers=12)
        print("创建了自定义模型")
else:
    # 使用自定义模型
    model = SimpleModel(vocab_size=50257, hidden_size=768, num_layers=12)
    print("创建了自定义模型")

print("应用LoRA...")
model = add_lora_to_gpt2(model, rank=8, alpha=16.0)

# 计算可训练参数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数: {all_params:,}")
print(f"可训练参数: {trainable_params:,} ({trainable_params/all_params:.2%} of all parameters)")
print(f"LoRA压缩率: {all_params/trainable_params:.1f}x")
