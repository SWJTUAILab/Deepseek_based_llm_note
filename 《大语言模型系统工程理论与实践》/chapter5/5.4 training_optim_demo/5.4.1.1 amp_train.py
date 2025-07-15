# PyTorch自动混合精度训练（AMP）验证代码
# 运行方法：python '5.4.1.1 amp_train.py'
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
    def forward(self, x):
        return self.linear(x)

def loss_function(output, target):
    return ((output - target) ** 2).mean()

def main():
    model = SimpleModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    x = torch.randn(8, 8).cuda()
    y = torch.randn(8, 8).cuda()
    for _ in range(2):
        with autocast():
            output = model(x)
            loss = loss_function(output, y)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print('AMP混合精度训练验证完成')

if __name__ == '__main__':
    main() 