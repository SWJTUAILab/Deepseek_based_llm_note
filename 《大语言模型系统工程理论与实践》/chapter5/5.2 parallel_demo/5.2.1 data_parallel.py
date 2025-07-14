# 数据并行验证代码
# 运行方法：
# torchrun --standalone --nnodes=1 --nproc_per_node=2 data_parallel.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

def setup():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)

def main():
    setup()
    # 构造简单数据
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

    model = SimpleModel().cuda()
    model = DDP(model, device_ids=[dist.get_rank() % torch.cuda.device_count()])
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(2):
        for xb, yb in dataloader:
            xb, yb = xb.cuda(), yb.cuda()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if dist.get_rank() == 0:
            print(f"Epoch {epoch} done.")
    cleanup()

if __name__ == '__main__':
    main() 