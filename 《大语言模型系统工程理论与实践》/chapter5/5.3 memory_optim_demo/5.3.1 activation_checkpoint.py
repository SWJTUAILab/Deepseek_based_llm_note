# 激活值重计算验证代码
# 运行方法：python activation_checkpoint.py
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn

class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(16, 16) for _ in range(6)
        ])
        self.activation = nn.ReLU()
    def forward(self, x):
        # 分2段，每段3层
        segments = [[self.layers[i+j] for j in range(3)] for i in range(0, 6, 3)]
        for segment in segments:
            x = checkpoint(self._segment_forward, x, segment)
        return x
    def _segment_forward(self, x, layers):
        for layer in layers:
            x = layer(x)
            x = self.activation(x)
        return x

def main():
    model = CheckpointedModel()
    x = torch.randn(4, 16, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print('激活值重计算前向和反向传播完成')

if __name__ == '__main__':
    main() 