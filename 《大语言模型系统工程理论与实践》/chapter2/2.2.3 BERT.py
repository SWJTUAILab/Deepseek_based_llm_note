from transformers import BertTokenizer, BertModel
import torch
# # 加载预训练的BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained("D:\Bert-base-uncased") #从hf-mirror中手动下载模型文件到D盘，绕开huggingface的代理限制，故路径有所更改
model = BertModel.from_pretrained("D:\Bert-base-uncased")
text = "BERT provides contextual embeddings."
# # 编码文本
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
# # 获取BERT模型的输出
with torch.no_grad():
    outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
# # last_hidden_states 的形状是 (batch_size, sequence_length, hidden_size)
# # 这可以被视为BERT生成的动态上下文词嵌入
print("Shape of BERT last hidden states:", last_hidden_states.shape)
