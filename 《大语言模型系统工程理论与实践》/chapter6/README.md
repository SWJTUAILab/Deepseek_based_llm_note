# 第六章代码验证

## 系统操作环境

  * **操作系统** ：Linux
  * **CPU** ：Intel Xeon Gold 6142
  * **GPU** ：NVIDIA GeForce RTX 3090

## 依赖安装

  1. 安装 PyTorch 及其配套的 CUDA（版本要合适，二者相匹配，比如 torch 2.2.2 搭配 CUDA 12.3）
     * `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121`

  2. 安装 NCCL 2.19.3
     * 需登录 NVIDIA 官网下载（注册账号后访问），下载链接示例：<https://developer.nvidia.com/nccl/nccl> \-download，选择与 CUDA 12.3 兼容的版本（nccl_2.19.3-1+cuda12.3_x86_64.txz）解压并安装。
     * `tar -xvf nccl_2.19.3-1+cuda12.3_x86_64.txz`
     * `sudo cp -r nccl_2.19.3-1+cuda12.3_x86_64/{lib,include} /usr/local/cuda-12.3/`

  3. 安装 fastmoe
     * 在 github 上找到并下载：<https://github.com/laekov/fastmoe.git>
     * `git clone https://github.com/laekov/fastmoe.git`
     * `cd fastmoe`
     * `export CUDA_HOME=/usr/local/cuda-12.3` （指向您的 CUDA 路径）
     * `python setup.py instal`

  4. 安装 deepspeed
     * 可通过预编译包安装：`pip install deepspeed`

  5. 安装 dm-tree
     * `pip install dm-tree`

## 代码及运行结果解释

### 6.4.1 fastmoe 的核心框架

  * **解释** ：此段代码为 fastmoe 中的混合专家模型的核心代码。它通过 `nn.ModuleList` 动态创建多个专家模块。根据 gate 参数初始化路由决策模块，采用 NaiveGate 计算每个 Token 的专家分配权重。门控路由决策中 `gate_top_k_idx` 表示每个 Token 选择的 top_k 个专家编号，`gate_score` 表示对应专家的权重分数。专家计算时，按 `gate_top_k_idx` 将输入 Token 分发到对应专家，每个专家独立处理分配到的 Token（并行执行），最后按 `gate_score` 加权合并专家输出。

### 6.4.3 和 6.4.4 在 Transformer 模型中集成 fastmoe 并训练

  * **依赖** ：torch2.2.2，CUDA12.3，NCCL2.19.3，dm-tree，fastmoe
  * **解释** ：这段代码展示了一个基于 fastmoe 的 Transformer 层的实现和训练，并在随机数据上进行了简单的训练和测试。训练结果显示损失下降，模型参数规模较大。模型参数总数约 2.73 亿，其中 MoE 层约 2.68 亿参数，注意力层约 5.24M 参数，归一化层约 0.2M 参数。输入数据范围 [-4.114,4.577]，标签数据范围 [-4.417,4.294]，数据分布符合标准正态分布特点，适合神经网络学习。训练损失从 2.04 下降到 1.11（下降 45%），验证模型结构合理。

### 6.4.2 deepspeed 核心框架

  * **解释** ：DeepSpeed 的 MoE 实现通过多种创新技术大幅优化了混合专家模型的性能和扩展性。其 moe 层使用了动态容量控制，参数包括训练容量系数、推理容量系数、最小处理单元数等。专家架构为金字塔式，底层大量小型专家，中层中等专家，顶层少量大型专家，适应不同复杂度 token。若 `use_residual`，则有残差专家机制兜底处理。门控层采用 TopKGate 机制，有注入噪声的门控策略（如添加高斯噪声或采用 Gumbel-Softmax 重参数化），可助于达到负载均衡的效果，同时采用弹性容量策略。

### 6.4.3 和 6.4.4 在 Transformer 模型中集成 deepspeed 并训练

  * **依赖** ：torch2.2.2，CUDA12.3，NCCL2.19.3，dm-tree，deepspeed
  * **解释** ：此代码展示了如何使用 DeepSpeed 训练 MoE 模型。训练结果显示损失稳定下降，证明了这种方法的可行性。DeepSpeed 的 ZeRO 和 MoE 支持使大规模模型训练更高效，适用于资源受限场景。在 MoE Transformer 层结构中，有隐藏层维度、专家总数、每个 token 选择专家数、专家容量缩放因子等参数。输入向量为生成的三维向量，测试输入形状和测试输出形状均为（32，8，512）。训练损失在模拟训练中，从 Epoch 1 的平均损失 1.784354 降至 Epoch 5 的平均损失 1.000710，明显下降。

## 问题记录

  1. 在下载 fastmoe 时，需进行 `python setup.py instal`，默认情况下分布式专家功能启用。若想禁用，需设置环境变量 `USE_NCCL=0` 传递给 Setup 脚本。若启用分布式专家功能，则使用 P2P 通信的 NCCL 需支持，通常是 versions >=2.7.5。
  2. 注意 cuda 与 torch 版本的适配性，若不适配，在编译 `python setup.py instal` 时会报错找不到文件。
  3. 在下载 deepspeed 时，需额外下载 dm-tree 包，否则使用 deepspeed 会报错。
  4. 总专家数量 num_experts 和 worker 总数 world_size 需根据实际硬件算力调整，若数量过多易导致卡死。
