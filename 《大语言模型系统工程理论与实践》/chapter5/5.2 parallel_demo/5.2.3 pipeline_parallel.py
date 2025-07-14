# 流水线并行验证代码
# 运行方法：
# deepspeed pipeline_parallel.py --deepspeed --deepspeed_config ds_config.json

import torch
import torch.nn as nn
from deepspeed.pipe import PipelineModule
import deepspeed

class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 20)
    def forward(self, x):
        return torch.relu(self.layer(x))

class Stage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(20, 1)
    def forward(self, x):
        return self.layer(x)

layers = [Stage1(), Stage2()]

model = PipelineModule(
    layers=layers,
    loss_fn=nn.MSELoss(),
    num_stages=2,
    partition_method='uniform',
    activation_checkpoint_interval=0
)

ds_config = {
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 4,
    "fp16": {"enabled": False},
    "pipeline": {"stages": 2}
}

with open('parallel_demo/ds_config.json', 'w') as f:
    import json
    json.dump(ds_config, f, indent=2)

engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)

x = torch.randn(8, 10)
y = torch.randn(8, 1)
loss = engine(x, y)
engine.backward(loss)
engine.step()
print('流水线并行前向和反向传播完成') 