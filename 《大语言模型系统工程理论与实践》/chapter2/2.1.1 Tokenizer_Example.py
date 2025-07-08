from transformers import AutoTokenizer

# 加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained("D:\Bert-base-cased")
                                         
# 使用分词器处理文本
encoded_input = tokenizer("Using a Transformer network is simple")
print(encoded_input)
# 输出示例:
# {"input_ids": [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
#  "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0],
#  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1]}

# 将输入ID解码回文本
decoded_output = tokenizer.decode(encoded_input["input_ids"])
print(decoded_output)
# 输出示例:
# "[CLS] Using a Transformer network is simple [SEP]"
