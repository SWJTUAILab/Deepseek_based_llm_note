# 训练优化相关验证代码说明
amp_train.py —— PyTorch自动混合精度训练（AMP）
deepspeed_amp.py —— DeepSpeed混合精度训练
bf16_train.py —— PyTorch BF16精度训练
jit_fusion.py —— PyTorch JIT融合
grad_accum.py —— PyTorch梯度累积
amp_grad_accum.py —— AMP+梯度累积
deepspeed_grad_accum.py —— DeepSpeed梯度累积
grad_compress.py —— PyTorch梯度压缩（简化版）
overlap_comm.py —— PyTorch通信与计算重叠（结构演示）
deepspeed_comm_optim.py —— DeepSpeed通信优化
requirements.txt —— 依赖说明
readme.md —— 详细运行说明
## 依赖安装

建议新建虚拟环境后：
```
pip install -r requirements.txt
```

## 1. 混合精度训练
- amp_train.py —— PyTorch自动混合精度训练（AMP）

![image-20250714234341183](images/image-20250714234341183.png)

- deepspeed_amp.py —— DeepSpeed混合精度训练

![image-20250714234356835](images/image-20250714234356835.png)

- bf16_train.py —— PyTorch BF16精度训练

![image-20250714234410149](images/image-20250714234410149.png)

## 2. 算子融合
- jit_fusion.py —— PyTorch JIT融合

![image-20250714234417860](images/image-20250714234417860.png)

## 3. 梯度累积
- grad_accum.py —— PyTorch梯度累积

![image-20250714234428587](images/image-20250714234428587.png)

- amp_grad_accum.py —— AMP+梯度累积

![image-20250714234433554](images/image-20250714234433554.png)

- deepspeed_grad_accum.py —— DeepSpeed梯度累积

![image-20250714234438339](images/image-20250714234438339.png)

## 4. 通信优化
- grad_compress.py —— PyTorch梯度压缩（简化版）

![image-20250714234444751](images/image-20250714234444751.png)

- overlap_comm.py —— PyTorch通信与计算重叠（结构演示）

![image-20250714234450930](images/image-20250714234450930.png)

- deepspeed_comm_optim.py —— DeepSpeed通信优化

![image-20250714234456427](images/image-20250714234456427.png)

---

每个脚本头部有运行方法，部分DeepSpeed脚本需配合ds_config.json。
如需更复杂的验证，可参考官方文档或进一步扩展。 