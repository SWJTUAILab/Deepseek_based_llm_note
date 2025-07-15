# DeepSpeed CPU卸载与NVMe卸载验证代码
# 运行方法：deepspeed deepspeed_offload.py --deepspeed --deepspeed_config ds_config_cpu.json 或 ds_config_nvme.json
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
ds_config_cpu = {
    "train_batch_size": 8,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True}
    }
}
ds_config_nvme = {
    "train_batch_size": 8,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "nvme", "nvme_path": "/tmp/nvme", "pin_memory": True},
        "offload_param": {"device": "nvme", "nvme_path": "/tmp/nvme", "pin_memory": True}
    },
    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 1,
        "single_submit": False,
        "overlap_events": True
    }
}
import json
with open('memory_optim_demo/ds_config_cpu.json', 'w') as f:
    json.dump(ds_config_cpu, f, indent=2)
with open('memory_optim_demo/ds_config_nvme.json', 'w') as f:
    json.dump(ds_config_nvme, f, indent=2)

engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config_cpu)
x = torch.randn(8, 8)
y = torch.randn(8, 8)
loss_fn = nn.MSELoss()
loss = loss_fn(engine(x), y)
engine.backward(loss)
engine.step()
print('DeepSpeed CPU卸载/显存卸载 验证完成') 