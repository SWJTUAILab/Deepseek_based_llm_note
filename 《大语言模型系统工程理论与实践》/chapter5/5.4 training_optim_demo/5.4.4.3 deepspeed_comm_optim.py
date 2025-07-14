# DeepSpeed通信优化验证代码
# 运行方法：deepspeed '5.4.4.3 deepspeed_comm_optim.py' --deepspeed --deepspeed_config ds_config.json
import torch
import torch.nn as nn
import deepspeed

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
    def forward(self, x):
        return self.linear(x)

def loss_function(output, target):
    return ((output - target) ** 2).mean()

model = SimpleModel()
ds_config = {
    "train_batch_size": 8,
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2},
    "communication_data_type": "fp16",
    "reduce_bucket_size": 5e7,
    "overlap_comm": True,
    "allgather_bucket_size": 5e7,
    "reduce_scatter": True
}
with open('training_optim_demo/ds_config.json', 'w') as f:
    import json
    json.dump(ds_config, f, indent=2)

engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
x = torch.randn(8, 8)
y = torch.randn(8, 8)
loss = loss_function(engine(x), y)
engine.backward(loss)
engine.step()
print('DeepSpeed通信优化验证完成') 