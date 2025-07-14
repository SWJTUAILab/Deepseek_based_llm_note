# PyTorch通信与计算重叠验证代码（简化版）
# 运行方法：python '5.4.4.2 overlap_comm.py'
import torch
import torch.nn as nn
import torch.distributed as dist

def main():
    # 仅作结构演示，实际需torchrun分布式环境
    # dist.init_process_group(backend="nccl")
    model = nn.Linear(8, 8).cuda()
    x = torch.randn(2, 8).cuda()
    y = torch.randn(2, 8).cuda()
    output = model(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    # 假设分两块
    chunk1 = grads[:1]
    chunk2 = grads[1:]
    # 创建流
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    # 异步all-reduce（伪代码，实际需分布式）
    with torch.cuda.stream(stream1):
        flat1 = chunk1[0].view(-1)
        # dist.all_reduce(flat1, async_op=True)
    with torch.cuda.stream(stream2):
        flat2 = chunk2[0].view(-1)
        # dist.all_reduce(flat2, async_op=True)
    print('通信与计算重叠结构演示完成')

if __name__ == '__main__':
    main() 