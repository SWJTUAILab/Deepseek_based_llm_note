from transformers import GPT2Tokenizer, GPT2Model
import torch
# 加载预训练的GPT-2分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained("D:\gpt2")
model = GPT2Model.from_pretrained("D:\gpt2")       # 使用GPT2Model获取隐藏状态作为嵌入
# 添加填充标记 (如果词汇表中没有)
if tokenizer.pad_token is None:
       tokenizer.add_special_tokens({"pad_token": "[PAD]"}) 
       model.resize_token_embeddings(len(tokenizer))      # 调整模型嵌入层大小
text = "Natural language processing is fascinating."
# 编码文本，并确保返回PyTorch张量，进行填充和截断
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
# 获取GPT模型的输出
# last_hidden_state 包含了序列中每个词元的最后一层隐藏状态
with torch.no_grad():     # 在推理模式下运行，不计算梯度
     outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
# last_hidden_states 的形状是 (batch_size, sequence_length, hidden_size)
# 这可以被视为GPT生成的动态上下文词嵌入
print("Shape of GPT last hidden states:", last_hidden_states.shape)
