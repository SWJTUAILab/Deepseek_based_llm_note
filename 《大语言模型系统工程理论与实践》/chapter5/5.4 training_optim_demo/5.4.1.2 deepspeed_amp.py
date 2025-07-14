# DeepSpeed混合精度训练验证代码
# 运行方法：deepspeed '5.4.1.2 deepspeed_amp.py' --deepspeed --deepspeed_config ds_config.json
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
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
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
print('DeepSpeed混合精度训练验证完成') 