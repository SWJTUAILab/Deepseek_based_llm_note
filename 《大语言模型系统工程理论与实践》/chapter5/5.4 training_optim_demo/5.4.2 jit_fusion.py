# PyTorch JIT融合验证代码
# 运行方法：python '5.4.2 jit_fusion.py'
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

fusion_module = FusionModule(8)
fused_module = torch.jit.script(fusion_module)

x = torch.randn(4, 8)
y = fused_module(x)
print('JIT融合输出:', y) 