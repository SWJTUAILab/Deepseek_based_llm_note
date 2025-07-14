# DeepSpeed FP16_Optimizer 验证代码
# 运行方法：deepspeed deepspeed_fp16_optimizer.py --deepspeed --deepspeed_config ds_config.json
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
    "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 16},
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8}},
    "zero_optimization": {"stage": 1}
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
print('DeepSpeed FP16_Optimizer 验证完成') 