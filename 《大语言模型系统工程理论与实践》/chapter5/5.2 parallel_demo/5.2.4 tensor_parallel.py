# 张量并行验证代码（简化版）
# 运行方法：
# torchrun --standalone --nnodes=1 --nproc_per_node=2 tensor_parallel.py

import torch
import torch.nn as nn
import torch.distributed as dist

def setup():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def cleanup():
    dist.destroy_process_group()

class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.out_per_part = output_size // self.world_size
        self.linear = nn.Linear(input_size, self.out_per_part).cuda(self.rank)
    def forward(self, x):
        x = x.cuda(self.rank)
        out = self.linear(x)
        gathered = [torch.zeros_like(out) for _ in range(self.world_size)]
        dist.all_gather(gathered, out)
        return torch.cat(gathered, dim=-1)

def main():
    setup()
    model = ColumnParallelLinear(10, 4)
    x = torch.randn(2, 10)
    y = model(x)
    if dist.get_rank() == 0:
        print('张量并行输出:', y)
    cleanup()

if __name__ == '__main__':
    main() 