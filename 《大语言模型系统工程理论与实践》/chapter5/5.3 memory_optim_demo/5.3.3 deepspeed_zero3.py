# DeepSpeed ZeRO-3 验证代码
# 运行方法：deepspeed deepspeed_zero3.py --deepspeed --deepspeed_config ds_config.json
import torch
import torch.nn as nn
import deepspeed

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
ds_config = {
    "train_batch_size": 8,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e7,
        "allgather_bucket_size": 5e7
    }
}
with open('memory_optim_demo/ds_config.json', 'w') as f:
    import json
    json.dump(ds_config, f, indent=2)

engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)

x = torch.randn(8, 8)
y = torch.randn(8, 8)
loss_fn = nn.MSELoss()
loss = loss_fn(engine(x), y)
engine.backward(loss)
engine.step()
print('DeepSpeed ZeRO-3 验证完成') 