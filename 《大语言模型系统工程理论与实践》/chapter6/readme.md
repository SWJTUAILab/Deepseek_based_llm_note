### 第六章代码验证

### 1系统操作环境
操作系统:Linux
CPU: Intel Xeon Gold 6142
GPU: NVIDIA GeForce RTX 3090

### 2需要安装的依赖以及代码、注释
#torch及其配套的CUDA(版本要合适,二者相匹配,比如torch 2.2.2搭配CUDA 12.3)pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2\-index-url https://download.pytorch.org/whl/cu121

#NCCL 2.19.3需登录 NVIDIA官网下载(注册账号后访问)下载链接示例:https://developer.nvidia.com/nccl/nccl-download选择与 CUDA 12.3兼容的版本(nccl_2.19.3-1+cuda12.3_x86_64.txz)解压并安装
tar-xvf nccl_2.19.3-1+cuda12.3_x86_64.txz sudo cp-r nccl_2.19.3-1+cuda12.3_x86_64/{lib,include}/usr/local/cuda-12.3/

#fastmoe在github上可以找到并下载
git clone https://github.com/laekov/fastmoe.git
cd fastmoe
export CUDA_HOME=/usr/local/cuda-12.3#指向您的CUDA路径
python setup.py instal

#deepspeed可通过预编译包安装
pip install deepspeed

#dm-tree
pip install dm-tree

### 3运行结果及解释
6.4.1fastmoe的核心框架
代码解释:此段代码为fastmoe中的混合专家模型的核心代码。它通过nn.ModuleList动态创建 num_expert个专家模块。根据 gate参数初始化路由决策模块，采用NaiveGate计算每个Token的专家分配权重。此段代码中的门控路由决策和moe专家计算是主要的核心。在门控路由决策中gate_top_k_idx表示每个Token选择的top_k个专家编号，gate_score表示对应专家的权重分数。在专家计算时，根据gate_top_k_idx将输入Token分发到对应专家，每个专家独立处理分配到的Token(并行执行)，最后按gate_score加权合并专家输出。

6.4.3-1在Transformer模型中集成fastmoe并训练。
依赖:torch2.2.2，CUDA12.3，NCCL2.19.3，dm-tree，fastmoe
代码解释:这段代码展示了一个基于fastmoe的Transformer层的实现和训练，并在随机数据上进行了简单的训练和测试。训练结果显示了损失下降，模型参数规模较大(主要是专家网络参数多)。混合专家结构中的(top_k=2)表示每个token选择前2个专家。num_experts代表专家总数。注意力和moe相结合，多头注意力处理序列依赖关系，MoE处理特征转换。训练数据使用生成的三维向量。模型参数方面，模型参数总数:273,031,296, MoE层:128个专家x(1024x1024+1024x1024)≈2.68亿参数，注意力层:4头x(1024x1024x3+1024x1024)≈5.24M参数归一化层:≈0.2M参数。输入数据范围:[-4.114,4.577]标签数据范围:[-4.417,4.294]，数据分布符合标准正态分布特点(均值0，标准差≈1)，适合神经网络学习。观察训练结果可以看到训练损失从2.04下降到1.11(下降45%)，验证模型结构合理。

6.4.2deepspeed核心框架
代码解释:DeepSpeed的MoE实现通过多种创新技术大幅优化了混合专家模型的性能
和扩展性。我们分别来看它的 moe层和他的门控机制层 TopKGate类。首先来看 moe层，它使用了动态容量控制其中: self.capacity factor表示训练容量系数self.eval_capacity_factor表示推理容量系数, self.min_capacity表示最小处理单元数。它的专家架构为金字塔式的创新架构。底层:大量小型专家(参数量≈100M)中层:中等专家(≈500M)顶层:少量大型专家(≈2B)形成金字塔结构,适应不同复杂度 token。if self.use_residual: self.residual_expert=nn.Linear(...)表示残差专家机制,当主流专家拒绝token时，残差专家兜底处理(类似路由故障保险)。再来看他的门控层，deepspeed采用 TopKGate的门控创新机制。采用注入噪声的门控策略。 if noisy_gate_policy=="Jitter":#添加高斯噪声(μ=0,σ=0.01)gate_logits+= torch.randn_like(gate_logits)* 0.01 elif noisy_gate_policy=="RSample":# Gumbel-Softmax重参数化
gate_logits=gumbel_softmax(gate_logits)可以助于达到负载均衡的效果。同时采用弹性容量策略,当 capacity_factor=1.0时:平均负载 100%,当 capacity_factor=1.2时:允许 20%过载,当 capacity_factor=0.8时:仅使用 80%容量(推理优化)。

### 6.4.3-2在Transformer模型中集成deepspeed并训练
依赖:torch2.2.2，CUDA12.3，NCCL2.19.3，dm-tree，deepspeed
代码解释:此代码展示了如何使用DeepSpeed训练MoE模型。图片中的训练结果证明了这种方法的可行性,损失稳定下降。DeepSpeed的 ZeRO和 MoE支持使大规模模型训练更高效，适用于资源受限场景。在MoE Transformer层结构中，hidden_size表示隐藏层维度，num_experts表示专家总数，k表示每个token选择专家数
capacity_factor表示专家容量缩放因子。输入向量为生成的三维向量，测试输入形状和测试输出形状均为(32，8，512).。训练损失选择MSE均方误差，在模拟训练中，
观察发现Epoch 1:平均损失1.784354→Epoch 5:平均损失1.000710。训练损失明显下降。

### 4问题记录
1.在下载fastmoe时，需要进行python setup.py instal。而默认情况下，分布式专家功能处于启用状态。如果您想禁用 it需要将环境变量传递给 Setup脚本，即设置USE_NCCL=0。而如果启用了分布式专家功能，则使用P2P通信的 NCCL需要support,通常是 versions>=2.7.5。
2需注意cuda与torch版本的适配性，如果cuda和torch版本不适配在编译行python setup.py instal时会报错找不到文件。
3.在下载deepspeed时，需要额外下载一个dm-tree的包，否则在使用deepspeed时会报错。
4.总专家数量num_experts和worker总数world_size需要根据实际硬件算力调整，如果数量过多容易导致卡死。
