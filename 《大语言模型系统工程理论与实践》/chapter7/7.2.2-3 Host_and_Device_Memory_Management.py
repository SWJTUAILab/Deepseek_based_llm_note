# 使用CUDA流水线减少数据传输开销
import torch
import time


def pipelined_inference(model, input_batches):
    results = []
    streams = [torch.cuda.Stream() for _ in range(2)]

    # 预热
    with torch.cuda.stream(streams[0]):
        input_batches[0] = input_batches[0].cuda(non_blocking=True)

    for i in range(len(input_batches) - 1):
        # 当前批次计算
        with torch.cuda.stream(streams[i % 2]):
            results.append(model(input_batches[i]))

        # 下一批次数据传输
        with torch.cuda.stream(streams[(i + 1) % 2]):
            input_batches[i + 1] = input_batches[i + 1].cuda(non_blocking=True)

    # 处理最后一个批次
    with torch.cuda.stream(streams[(len(input_batches) - 1) % 2]):
        results.append(model(input_batches[-1]))

    # 同步所有流
    torch.cuda.synchronize()

    return results


# 程序调用
# 简单模型（可替换为其他训练好的模型）
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)


# 创建模型并转移到 GPU
model = SimpleModel().cuda()
model.eval()
# 构造模拟输入数据（例如总共128个样本，批大小为32）
batch_size = 32
total_samples = 128
input_data = torch.randn(total_samples, 1024)

# 划分成多个批次
input_batches = [input_data[i:i + batch_size] for i in range(0, total_samples, batch_size)]
with torch.no_grad():
    start = time.time()
    results = pipelined_inference(model, input_batches)
    end = time.time()
# 输出结果信息
print(f"推理总耗时: {end - start:.4f} 秒")
print(f"总批次数: {len(results)}")
print(f"第一批输出形状: {results[0].shape}")
