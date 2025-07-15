import numpy as np
import torch
import torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """初始化位置编辑器，d_model为模型的维度， max_len是最大序列长度"""
        super(PositionalEncoding, self).__init__()
       #创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        #创建索引矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #按照公式计算角度缩放因子，div_term = 10000^(-i/d_model)，i是维度索引
        #torch.arange(0, d_model, 2)生成从0开始，步长为2的张量，偶数维度的索引
        #np.log(10000)是自然对数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        #偶数阶维度用正弦函数，奇数阶维度用余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #添加批次维度，以便与输入的张量匹配
        pe = pe.unsqueeze(0) # shape (1, max_len, d_model)
        self.register_buffer("pe", pe) # 将位置编码注册为模型的缓冲区，不参与模型参数更新，但随模型一起保存
    def forward(self, x):
        """将位置编码添加到输入张量，x为输入的张量，形状为[batch_size, seq_len, d_model]"""
        #x shape: (batch_size, seq_len, d_model)
        #self.pe[:, :x.size(1), :] shape: (1, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].detach() # detach以确保pe不参与梯度计算
        return x





 # 示例使用
 # d_model = 512
 # max_seq_len = 100
 # pos_encoder = PositionalEncoding(d_model, max_seq_len)
 # input_embeddings = torch.randn(32, 60, d_model) # batch_size=32, seq_len=60
 # output_with_pe = pos_encoder(input_embeddings)
 # print("Shape after adding positional encoding:", output_with_pe.shape)

# import numpy as np
# import torch
# import torch.nn as nn

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0) # shape (1, max_len, d_model)
#         self.register_buffer("pe", pe) # 不参与模型参数更新

#      def forward(self, x):
#         x shape: (batch_size, seq_len, d_model)
#         # self.pe[:, :x.size(1), :] shape: (1, seq_len, d_model)
#         x = x + self.pe[:, :x.size(1), :].detach() # detach以确保pe不参与梯度计算
#         return x

# # 示例使用
# # d_model = 512
# # max_seq_len = 100
# # pos_encoder = PositionalEncoding(d_model, max_seq_len)
# # input_embeddings = torch.randn(32, 60, d_model) # batch_size=32, seq_len=60
# # output_with_pe = pos_encoder(input_embeddings)
# # print("Shape after adding positional encoding:", output_with_pe.shape)
