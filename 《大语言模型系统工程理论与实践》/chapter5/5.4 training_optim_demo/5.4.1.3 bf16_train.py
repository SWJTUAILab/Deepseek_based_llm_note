# PyTorch BF16精度训练验证代码
# 运行方法：python '5.4.1.3 bf16_train.py'
import torch
from torch.cuda.amp import autocast
import torch.nn as nn

def main():
    assert torch.cuda.is_bf16_supported(), '当前硬件不支持BF16'
    model = nn.Linear(8, 8).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, 8).cuda()
    y = torch.randn(8, 8).cuda()
    for _ in range(2):
        with autocast(dtype=torch.bfloat16):
            output = model(x)
            loss = ((output - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('BF16精度训练验证完成')

if __name__ == '__main__':
    main() 