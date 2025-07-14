# PyTorch梯度累积验证代码
# 运行方法：python '5.4.3.1 grad_accum.py'
import torch
import torch.nn as nn

def loss_function(output, target):
    return ((output - target) ** 2).mean()

model = nn.Linear(8, 8).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
accumulation_steps = 4
x = torch.randn(16, 8).cuda()
y = torch.randn(16, 8).cuda()

for i in range(accumulation_steps):
    xb = x[i*2:(i+1)*2]
    yb = y[i*2:(i+1)*2]
    output = model(xb)
    loss = loss_function(output, yb) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
print('梯度累积验证完成') 