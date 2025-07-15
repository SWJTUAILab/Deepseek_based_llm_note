# 模型并行验证代码
# 运行方法：
# CUDA_VISIBLE_DEVICES=0,1 python model_parallel.py

import torch
import torch.nn as nn

class ModelParallelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(nn.Linear(10, 20), nn.ReLU()).to('cuda:0')
        self.seq2 = nn.Sequential(nn.Linear(20, 1)).to('cuda:1')
    def forward(self, x):
        x = x.to('cuda:0')
        x = self.seq1(x)
        x = x.to('cuda:1')
        x = self.seq2(x)
        return x

def main():
    model = ModelParallelNet()
    x = torch.randn(16, 10)
    y = torch.randn(16, 1).to('cuda:1')
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    print('模型并行前向和反向传播完成')

if __name__ == '__main__':
    main() 