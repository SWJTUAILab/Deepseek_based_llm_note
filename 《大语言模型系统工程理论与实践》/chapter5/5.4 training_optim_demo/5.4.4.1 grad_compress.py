# PyTorch梯度压缩验证代码（简化版）
# 运行方法：python '5.4.4.1 grad_compress.py'
import torch
import torch.nn as nn

def compress(grad, ratio=0.5):
    k = max(1, int(grad.numel() * ratio))
    values, indices = torch.topk(grad.abs().view(-1), k)
    sparse = torch.zeros_like(grad.view(-1))
    sparse[indices] = grad.view(-1)[indices]
    return indices, sparse[indices], grad.shape

def decompress(indices, values, shape):
    grad = torch.zeros(shape).view(-1)
    grad[indices] = values
    return grad.view(shape)

model = nn.Linear(8, 8)
x = torch.randn(2, 8)
y = torch.randn(2, 8)
output = model(x)
loss = ((output - y) ** 2).mean()
loss.backward()
for param in model.parameters():
    if param.grad is not None:
        indices, values, shape = compress(param.grad, 0.5)
        recovered = decompress(indices, values, shape)
        print('原始梯度:', param.grad)
        print('压缩后恢复梯度:', recovered)
print('梯度压缩验证完成') 