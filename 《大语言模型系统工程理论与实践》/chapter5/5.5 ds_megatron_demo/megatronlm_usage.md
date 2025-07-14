# Megatron-LM 框架实践说明

## 1. 安装
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

## 2. 数据准备
- 使用 tools/preprocess_data.py 脚本将原始文本转为 .bin/.idx 格式
- 需准备 vocab.json、merges.txt

## 3. 典型训练命令
```bash
python pretrain_gpt.py \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size 4 \
    --global-batch-size 1024 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints \
    --data-path /path/to/dataset_idx_file \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --fp16 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 4
```

## 4. 关键参数说明
- --tensor-model-parallel-size: 张量并行度
- --pipeline-model-parallel-size: 流水线并行度
- --num-layers, --hidden-size, --num-attention-heads: 模型结构
- --micro-batch-size, --global-batch-size: 批次设置
- --fp16 或 --bf16: 混合精度
- --use-flash-attn: 启用FlashAttention

## 5. 训练流程
- 入口脚本如 pretrain_gpt.py，自动初始化分布式、加载数据、构建模型、训练与保存

## 6. 参考
- [Megatron-LM官方文档](https://github.com/NVIDIA/Megatron-LM) 