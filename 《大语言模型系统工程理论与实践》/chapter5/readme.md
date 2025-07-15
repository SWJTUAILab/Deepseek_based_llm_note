# Chapter 5 大语言模型系统工程实践汇总

本章包含大语言模型系统工程中的并行训练、内存优化、训练优化及DeepSpeed/Megatron-LM框架实践等多个方面的验证代码与说明。各子目录内容如下：

---

## 5.2 各类并行验证（parallel_demo）

- **data_parallel.py**：数据并行（PyTorch DDP）
- **model_parallel.py**：模型并行（模型不同部分在不同GPU）
- **pipeline_parallel.py**：流水线并行（DeepSpeed PipelineModule）
- **tensor_parallel.py**：张量并行（简化版，线性层输出多卡拼接）
- **hybrid_parallel.py**：混合并行（数据并行+模型并行）

**依赖安装**：
```bash
pip install -r 5.2\ parallel_demo/requirements.txt
```

**运行方法**：
- 数据并行：
  ```bash
  torchrun --standalone --nnodes=1 --nproc_per_node=2 data_parallel.py
  ```
- 模型并行：
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python model_parallel.py
  ```
- 流水线并行：
  ```bash
  deepspeed pipeline_parallel.py --deepspeed --deepspeed_config ds_config.json
  ```
- 张量并行：
  ```bash
  torchrun --standalone --nnodes=1 --nproc_per_node=2 tensor_parallel.py
  ```
- 混合并行：
  ```bash
  torchrun --standalone --nnodes=1 --nproc_per_node=2 hybrid_parallel.py
  ```

> 说明：如需更复杂的张量/流水线/混合并行，可参考Megatron-LM等库。

---

## 5.3 内存优化与激活值重计算（memory_optim_demo）

- **activation_checkpoint.py**：PyTorch激活值重计算（checkpoint）
- **deepspeed_fp16_optimizer.py**：DeepSpeed FP16_Optimizer 验证
- **bnb_8bit_optimizer.py**：bitsandbytes 8-bit Adam 验证
- **deepspeed_zero3.py**：DeepSpeed ZeRO-3 验证
- **deepspeed_offload.py**：DeepSpeed CPU卸载/NVMe卸载
- **custom_activation_offload.py**：自定义激活值卸载

**依赖安装**：
```bash
pip install -r 5.3\ memory_optim_demo/requirements.txt
```

**运行方法**：
- 激活值重计算：
  ```bash
  python '5.3.1 activation_checkpoint.py'
  ```
- DeepSpeed FP16_Optimizer：
  ```bash
  deepspeed '5.3.2 deepspeed_fp16_optimizer.py' --deepspeed --deepspeed_config ds_config.json
  ```
- DeepSpeed ZeRO-3：
  ```bash
  deepspeed '5.3.3 deepspeed_zero3.py' --deepspeed --deepspeed_config ds_config.json
  ```
- DeepSpeed CPU/NVMe卸载：
  ```bash
  deepspeed '5.3.4 deepspeed_offload.py' --deepspeed --deepspeed_config ds_config.json
  # 或
  deepspeed '5.3.4 deepspeed_offload.py' --deepspeed --deepspeed_config ds_config_nvme.json
  ```

> 说明：如需更复杂的验证，可参考官方文档或进一步扩展。

---

## 5.4 训练优化相关（training_optim_demo）

- **amp_train.py**：PyTorch自动混合精度训练（AMP）
- **deepspeed_amp.py**：DeepSpeed混合精度训练
- **bf16_train.py**：PyTorch BF16精度训练
- **jit_fusion.py**：PyTorch JIT融合
- **grad_accum.py**：PyTorch梯度累积
- **amp_grad_accum.py**：AMP+梯度累积
- **deepspeed_grad_accum.py**：DeepSpeed梯度累积
- **grad_compress.py**：PyTorch梯度压缩（简化版）
- **overlap_comm.py**：PyTorch通信与计算重叠
- **deepspeed_comm_optim.py**：DeepSpeed通信优化

**依赖安装**：
```bash
pip install -r 5.4\ training_optim_demo/requirements.txt
```

**主要功能分类**：
1. 混合精度训练（AMP/BF16/DeepSpeed）
2. 算子融合（JIT）
3. 梯度累积（PyTorch/AMP/DeepSpeed）
4. 通信优化（梯度压缩、通信与计算重叠、DeepSpeed通信优化）

> 说明：每个脚本头部有运行方法，部分DeepSpeed脚本需配合ds_config.json。如需更复杂的验证，可参考官方文档或进一步扩展。

---

## 5.5 DeepSpeed & Megatron-LM 框架实践（ds_megatron_demo）

- **ds_config.json**：典型DeepSpeed配置文件
- **deepspeed_train.py**：DeepSpeed训练主流程验证
- **megatronlm_usage.md**：Megatron-LM实践命令与说明

**依赖安装**：
```bash
pip install -r 5.5\ ds_megatron_demo/requirements.txt
```

**运行方法**：
- DeepSpeed训练：
  ```bash
  deepspeed deepspeed_train.py --deepspeed --deepspeed_config ds_config.json
  ```
- Megatron-LM：
  需源码安装，详见`megatronlm_usage.md`

> 说明：如需更复杂的分布式/并行/混合精度/ZeRO等功能，可参考官方文档和本目录示例。

---

## 总结

本章各子目录分别覆盖了大模型工程中的并行训练、内存优化、训练优化及主流分布式训练框架的实践。建议根据实际需求选择相应子目录进行实验和学习，所有脚本均配有依赖说明和运行方法，便于快速上手。 