# DeepSpeed训练主流程验证代码
# 运行方法：deepspeed '5.5.1 deepspeed_train.py' --deepspeed --deepspeed_config ds_config.json
import torch
import torch.nn as nn
import deepspeed
import os

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
    def forward(self, x):
        return self.linear(x)

def loss_function(output, target):
    return ((output - target) ** 2).mean()

def to_device(batch, device):
    return batch.to(device)

def main():
    model = SimpleModel()
    ds_config = 'ds_config.json'
    # DeepSpeed会自动创建优化器和调度器
    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    for step in range(2):
        xb = to_device(x, engine.local_rank) if hasattr(engine, 'local_rank') else x
        yb = to_device(y, engine.local_rank) if hasattr(engine, 'local_rank') else y
        loss = engine(xb)
        engine.backward(loss_function(loss, yb))
        engine.step()
        print(f"step {step} done.")
    # 保存和加载checkpoint
    save_dir = './checkpoints'
    tag = 'demo'
    os.makedirs(save_dir, exist_ok=True)
    engine.save_checkpoint(save_dir, tag=tag)
    print('已保存checkpoint')
    engine.load_checkpoint(save_dir, tag=tag)
    print('已加载checkpoint')

if __name__ == '__main__':
    main() 