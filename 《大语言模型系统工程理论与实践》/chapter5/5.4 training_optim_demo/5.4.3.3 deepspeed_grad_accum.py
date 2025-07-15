# DeepSpeed梯度累积验证代码
# 运行方法：deepspeed '5.4.3.3 deepspeed_grad_accum.py' --deepspeed --deepspeed_config ds_config.json
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
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": True}
}
with open('training_optim_demo/ds_config.json', 'w') as f:
    import json
    json.dump(ds_config, f, indent=2)

engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
x = torch.randn(4, 8)
y = torch.randn(4, 8)
loss = loss_function(engine(x), y)
engine.backward(loss)
engine.step()
print('DeepSpeed梯度累积验证完成') 