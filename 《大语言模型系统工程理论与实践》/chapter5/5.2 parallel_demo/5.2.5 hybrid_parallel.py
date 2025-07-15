# 混合并行验证代码（数据并行+模型并行）
# 运行方法：
# torchrun --standalone --nnodes=1 --nproc_per_node=2 hybrid_parallel.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

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

def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def cleanup():
    dist.destroy_process_group()

def main():
    local_rank = setup()
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    dataset = TensorDataset(x, y)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

    model = ModelParallelNet()
    model = DDP(model, device_ids=[local_rank])
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