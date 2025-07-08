import torch
import torch.nn as nn
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):  #初始化位置编码器
        super(LearnedPositionalEncoding, self).__init__()  #定义的类继承自nn,Module类，用super()
        #创建表格，存储从位置索引到d_model维向量的映射；这个表格是要学习的位置信息。
        self.pos_embedding = nn.Embedding(max_len, d_model)  #nn.Embedding是pytorch中专门存储“索引→向量”的映射工具
        # 初始化位置编码 (可选，但通常有益)
        # nn.init.xavier_uniform_(self.pos_embedding.weight)
    #前向传播：为输入位置添加位置编码
    def forward(self, x):
        # x shape（形状）: (batch_size, seq_len, d_model)
        #获取输入序列长度
        seq_len = x.size(1)
        #创建位置索引张量，unsqueeze将其扩展为形状
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0) # shape (1, seq_len)
        positions = positions.expand(x.size(0), seq_len) # shape (batch_size, seq_len) # 可选，取决于Embedding层是否自动处理广播
        #通过嵌入曾获取每个位置的编码向量
        position_embeddings = self.pos_embedding(positions) # shape (1, seq_len, d_model) or (batch_size, seq_len, d_model)
        #将位置编码添加到输入张量
        x = x + position_embeddings
        return x

# # 示例使用
d_model = 512
max_seq_len = 100
learned_pos_encoder = LearnedPositionalEncoding(d_model, max_seq_len)
input_embeddings = torch.randn(32, 60, d_model)
output_with_learned_pe = learned_pos_encoder(input_embeddings)
print("Shape after adding learned positional encoding:", output_with_learned_pe.shape)
