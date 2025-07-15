from allennlp.modules.elmo import Elmo, batch_to_ids
import torch

# 加载预训练的ELMo模型配置和权重文件路径
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# 初始化ELMo模型，num_output_representations=1 表示只输出顶层加权和的表示
# 如果需要所有层的表示，可以设置为 L+1 (L是LSTM层数)
elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)

# 示例句子 (已经分好词)
sentences = [["I", "love", "natural", "language", "processing"],
            ["The", "bank", "is", "on", "the", "river", "bank"]]

# 将句子转换为字符ID序列
character_ids = batch_to_ids(sentences)

# 获取ELMo嵌入
# embeddings["elmo_representations"] 是一个列表，包含指定数量的输出表示
# embeddings["mask"] 是输入序列的掩码
embeddings = elmo(character_ids)
elmo_vectors = embeddings["elmo_representations"][0] # 获取第一个 (也是唯一的) 输出表示

# elmo_vectors 的形状是 (batch_size, sequence_length, embedding_dim)
print("Shape of ELMo vectors:", elmo_vectors.shape)
# 打印第一个句子中 "bank" 的向量 (如果存在)
# 注意：实际应用中需要根据分词结果定位词的索引
