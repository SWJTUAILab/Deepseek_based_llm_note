#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROCm实践示例代码
包含混合精度训练、分布式训练、性能优化等实际应用案例
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import os

# 检查ROCm环境
def check_rocm_environment():
    """检查ROCm环境配置"""
    print("=== ROCm环境检查 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"ROCm版本: {torch.version.hip}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f}GB")
    
    # 检查混合精度支持
    print(f"FP16支持: {torch.cuda.is_fp16_supported()}")
    print(f"BF16支持: {torch.cuda.is_bf16_supported()}")
    print()

# 示例1: 基础混合精度训练
def basic_mixed_precision_training():
    """基础混合精度训练示例"""
    print("=== 基础混合精度训练示例 ===")
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    ).cuda()
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建GradScaler用于混合精度训练
    scaler = amp.GradScaler()
    
    # 创建随机数据
    batch_size = 32
    input_data = torch.randn(batch_size, 1024).cuda()
    target_data = torch.randn(batch_size, 64).cuda()
    
    # 损失函数
    criterion = nn.MSELoss()
    
    print("开始混合精度训练...")
    start_time = time.time()
    
    for epoch in range(10):
        # 使用autocast进行混合精度前向传播
        with amp.autocast():
            output = model(input_data)
            loss = criterion(output, target_data)
        
        # 使用scaler进行反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.2f}秒")
    print()

# 示例2: 高级混合精度训练（带梯度累积）
def advanced_mixed_precision_training():
    """高级混合精度训练示例（带梯度累积）"""
    print("=== 高级混合精度训练示例 ===")
    
    # 创建更大的模型
    model = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 128)
    ).cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scaler = amp.GradScaler()
    criterion = nn.MSELoss()
    
    # 梯度累积步数
    accumulation_steps = 4
    batch_size = 16
    
    # 创建数据
    input_data = torch.randn(batch_size, 2048).cuda()
    target_data = torch.randn(batch_size, 128).cuda()
    
    print("开始高级混合精度训练（带梯度累积）...")
    start_time = time.time()
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        
        for step in range(accumulation_steps):
            with amp.autocast():
                output = model(input_data)
                loss = criterion(output, target_data) / accumulation_steps
            
            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            total_loss += loss.item()
            
            # 每accumulation_steps步更新一次参数
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        avg_loss = total_loss * accumulation_steps
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")
    
    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.2f}秒")
    print()

# 示例3: 性能对比测试
def performance_comparison():
    """FP32 vs FP16 vs BF16 性能对比"""
    print("=== 性能对比测试 ===")
    
    # 创建测试模型
    model_fp32 = nn.Sequential(
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    ).cuda()
    
    model_fp16 = model_fp32.half().cuda()
    
    # 创建测试数据
    batch_size = 64
    input_data_fp32 = torch.randn(batch_size, 4096).cuda()
    input_data_fp16 = input_data_fp32.half()
    
    # 预热
    for _ in range(10):
        _ = model_fp32(input_data_fp32)
        _ = model_fp16(input_data_fp16)
    
    torch.cuda.synchronize()
    
    # FP32测试
    start_time = time.time()
    for _ in range(100):
        _ = model_fp32(input_data_fp32)
    torch.cuda.synchronize()
    fp32_time = time.time() - start_time
    
    # FP16测试
    start_time = time.time()
    for _ in range(100):
        _ = model_fp16(input_data_fp16)
    torch.cuda.synchronize()
    fp16_time = time.time() - start_time
    
    print(f"FP32推理时间: {fp32_time:.4f}秒")
    print(f"FP16推理时间: {fp16_time:.4f}秒")
    print(f"FP16加速比: {fp32_time/fp16_time:.2f}x")
    print()

# 示例4: 内存使用监控
def memory_monitoring():
    """内存使用监控示例"""
    print("=== 内存使用监控 ===")
    
    def print_memory_usage(stage):
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
    
    # 初始状态
    print_memory_usage("初始状态")
    
    # 创建大模型
    model = nn.Sequential(
        nn.Linear(8192, 4096),
        nn.ReLU(),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    ).cuda()
    
    print_memory_usage("模型加载后")
    
    # 创建大批次数据
    batch_size = 128
    input_data = torch.randn(batch_size, 8192).cuda()
    
    print_memory_usage("数据加载后")
    
    # 前向传播
    with torch.no_grad():
        output = model(input_data)
    
    print_memory_usage("前向传播后")
    
    # 清理
    del model, input_data, output
    torch.cuda.empty_cache()
    
    print_memory_usage("清理后")
    print()

# 示例5: 分布式训练设置（单机多卡）
def setup_distributed_training(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed_training():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def distributed_training_worker(rank, world_size):
    """分布式训练工作进程"""
    setup_distributed_training(rank, world_size)
    
    # 创建模型
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    ).cuda(rank)
    
    # 包装为DDP模型
    model = DDP(model, device_ids=[rank])
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 创建数据（每个进程使用不同的数据）
    batch_size = 32
    input_data = torch.randn(batch_size, 1024).cuda(rank)
    target_data = torch.randn(batch_size, 128).cuda(rank)
    
    print(f"进程 {rank}: 开始分布式训练...")
    
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        
        if rank == 0 and epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    cleanup_distributed_training()

def run_distributed_training():
    """运行分布式训练示例"""
    print("=== 分布式训练示例 ===")
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("需要至少2个GPU来运行分布式训练")
        return
    
    print(f"使用 {world_size} 个GPU进行分布式训练")
    
    # 启动多进程
    mp.spawn(
        distributed_training_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    print()

# 示例6: 自定义性能分析
def performance_profiling():
    """性能分析示例"""
    print("=== 性能分析示例 ===")
    
    # 创建测试模型
    model = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    ).cuda()
    
    # 创建测试数据
    batch_size = 64
    input_data = torch.randn(batch_size, 2048).cuda()
    
    # 预热
    for _ in range(10):
        _ = model(input_data)
    
    torch.cuda.synchronize()
    
    # 记录开始时间
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 开始计时
    start_event.record()
    
    # 执行多次推理
    num_iterations = 100
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(input_data)
    
    # 结束计时
    end_event.record()
    torch.cuda.synchronize()
    
    # 计算时间
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
    avg_time = elapsed_time / num_iterations
    throughput = batch_size * num_iterations / elapsed_time
    
    print(f"总时间: {elapsed_time:.4f}秒")
    print(f"平均推理时间: {avg_time*1000:.2f}毫秒")
    print(f"吞吐量: {throughput:.0f} 样本/秒")
    print()

# 主函数
def main():
    """主函数"""
    print("ROCm实践示例代码")
    print("=" * 50)
    
    # 检查环境
    check_rocm_environment()
    
    # 运行各种示例
    try:
        basic_mixed_precision_training()
        advanced_mixed_precision_training()
        performance_comparison()
        memory_monitoring()
        performance_profiling()
        
        # 分布式训练需要多GPU
        if torch.cuda.device_count() >= 2:
            run_distributed_training()
        else:
            print("跳过分布式训练示例（需要多GPU）")
            
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请检查ROCm环境配置")

if __name__ == "__main__":
    main() 