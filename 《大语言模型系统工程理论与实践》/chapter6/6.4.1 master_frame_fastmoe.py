class FMoE(nn.Module):
    """
    һ��ͨ�õ�MoEʵ�֣�֧������ģ����Ϊר�ҡ�
    
    ����:
    - num_expert: ÿ��worker�ϵ�ר������
    - d_model: ��������ά��
    - world_size: ������ͬר�ҵ�worker����
    - slice_group: ͨ���飬ָʾ�ض���ģ�Ͳ���Ӧ���ڸ���
    - moe_group: ִ��ר�Ҳ��еĽ�����
    - top_k: ÿ��tokenҪ·�ɵ���ר������
    - gate: �ſ������࣬����fmoe.gates���ҵ�
    - expert: ר��ģ���࣬��������num_expert��ר��ģ��
    - gate_bias: �Ƿ����ſ�ģ�������ƫ��
    """
    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,  # ������
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
        # ����ͨ�����������
        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        
        # ������Ƭ��С������
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()
            
        self.top_k = top_k
        
        # ��ʼ��ר��ģ��
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True
            
        # ��ʼ���ſ�����
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
        MoEģ�����ȼ����ſ������Ȼ������ſؽ���MoEǰ����㡣
        ר�Ҹ�����ѡ���ŵķ��������ר�ҵ����������ΪȨ�ء�
        """
        # ����������δ�Сһ����
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"
        
        # ������豸ͨ��
        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)
            tree.map_structure(ensure_comm_func, moe_inp)
            
        # ������Ƭ
        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )
            moe_inp = tree.map_structure(slice_func, moe_inp)
            
        # �����ſ����
        gate_top_k_idx, gate_score = self.gate(moe_inp)
        
        # Ӧ���ſع��ӣ�����У�
        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)
            
        # �������루����У�
        if self.mask is not None and self.mask_dict is not None:
            # ɾ�����������Ĵ���...
            pass
            
        # ִ��MoEǰ�����
        fwd = _fmoe_general_global_forward(
            moe_inp,
            gate_top_k_idx,
            self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
            self.num_expert,
            self.world_size,
            experts=self.experts
        )
        
        # �ָ�ɾ��������������У�
        if self.mask is not None and self.mask_dict is not None:
            # �ָ�����...
            pass
            
        # �������
        if self.slice_size > 1:
            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )
            moe_outp = tree.map_structure(all_gather_func, fwd)
        else:
            moe_outp = fwd
            
        return moe_outp
FastMoE�ṩ�˶����ſ�����ʵ�֣����������NaiveGate�ࣺ
class NaiveGate(nn.Module):
    """
    һ���򵥵��ſ����磬ʹ�����Բ㽫����ӳ�䵽ר��ѡ����ʡ�
    
    ����:
    - d_model: ��������ά��
    - num_expert: ר������
    - world_size: ������ͬר�ҵ�worker����
    - top_k: ÿ��tokenҪ·�ɵ���ר������
    - gate_bias: �Ƿ����ſ�ģ�������ƫ��
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
        �����ſ������ѡ��top-kר�ҡ�
        
        ����:
        - gate_top_k_idx: ѡ�е�ר������
        - gate_score: ר�ҵ÷�
        """
        # ��ȡ����ĵ�һ������
        if isinstance(inp, tuple) or isinstance(inp, list):
            inp = inp[0]
            
        # �����ſص÷�
        gate_logit = self.gate(inp)
        
        # ѡ��top-kר��
        gate_score, gate_top_k_idx = torch.topk(gate_logit, k=self.top_k, dim=1)
        
        # Ӧ��softmax��һ��
        gate_score = self.softmax(gate_score)
        
        return gate_top_k_idx, gate_score