class FMoE(nn.Module):
    """
    一个通用的MoE实现，支持任意模块作为专家。
    
    参数:
    - num_expert: 每个worker上的专家数量
    - d_model: 输入特征维度
    - world_size: 包含不同专家的worker总数
    - slice_group: 通信组，指示特定的模型并行应用于该组
    - moe_group: 执行专家并行的进程组
    - top_k: 每个token要路由到的专家数量
    - gate: 门控网络类，可在fmoe.gates中找到
    - expert: 专家模块类，用于生成num_expert个专家模块
    - gate_bias: 是否在门控模块中添加偏置
    """
    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,  # 已弃用
        slice_group=None,
        moe_group=None,
        top_k=2,
        gate=NaiveGate,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
        gate_bias=True,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        # 处理通信组相关设置
        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        
        # 设置切片大小和排名
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()
            
        self.top_k = top_k
        
        # 初始化专家模块
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True
            
        # 初始化门控网络
        if issubclass(gate, NaiveGate):
            self.gate = gate(d_model, num_expert, world_size, top_k, gate_bias=gate_bias)
        else:
            self.gate = gate(d_model, num_expert, world_size, top_k)
            
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

    def forward(self, moe_inp):
        """
        MoE模块首先计算门控输出，然后根据门控进行MoE前向计算。
        专家给出的选定门的分数会乘以专家的输出张量作为权重。
        """
        # 检查输入批次大小一致性
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"
        
        # 处理多设备通信
        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)
            tree.map_structure(ensure_comm_func, moe_inp)
            
        # 处理切片
        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )
            moe_inp = tree.map_structure(slice_func, moe_inp)
            
        # 计算门控输出
        gate_top_k_idx, gate_score = self.gate(moe_inp)
        
        # 应用门控钩子（如果有）
        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)
            
        # 处理掩码（如果有）
        if self.mask is not None and self.mask_dict is not None:
            # 删除掩码张量的处理...
            pass
            
        # 执行MoE前向计算
        fwd = _fmoe_general_global_forward(
            moe_inp,
            gate_top_k_idx,
            self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
            self.num_expert,
            self.world_size,
            experts=self.experts
        )
        
        # 恢复删除的张量（如果有）
        if self.mask is not None and self.mask_dict is not None:
            # 恢复处理...
            pass
            
        # 处理输出
        if self.slice_size > 1:
            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )
            moe_outp = tree.map_structure(all_gather_func, fwd)
        else:
            moe_outp = fwd
            
        return moe_outp
FastMoE提供了多种门控网络实现，最基本的是NaiveGate类：
class NaiveGate(nn.Module):
    """
    一个简单的门控网络，使用线性层将输入映射到专家选择概率。
    
    参数:
    - d_model: 输入特征维度
    - num_expert: 专家总数
    - world_size: 包含不同专家的worker总数
    - top_k: 每个token要路由到的专家数量
    - gate_bias: 是否在门控模块中添加偏置
    """
    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
        super().__init__()
        self.gate = nn.Linear(d_model, num_expert * world_size, bias=gate_bias)
        self.top_k = top_k
        self.num_expert = num_expert
        self.world_size = world_size
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp):
        """
        计算门控输出，选择top-k专家。
        
        返回:
        - gate_top_k_idx: 选中的专家索引
        - gate_score: 专家得分
        """
        # 获取输入的第一个张量
        if isinstance(inp, tuple) or isinstance(inp, list):
            inp = inp[0]
            
        # 计算门控得分
        gate_logit = self.gate(inp)
        
        # 选择top-k专家
        gate_score, gate_top_k_idx = torch.topk(gate_logit, k=self.top_k, dim=1)
        
        # 应用softmax归一化
        gate_score = self.softmax(gate_score)
        
        return gate_top_k_idx, gate_score