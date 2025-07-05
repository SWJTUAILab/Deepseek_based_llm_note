# 使用ONNX Runtime和INT8量化部署小型LLM
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# 创建ONNX Runtime推理会话
ort_session = ort.InferenceSession(
    "distilgpt2_int8.onnx",
    providers=["CPUExecutionProvider"]  # 或使用"CUDAExecutionProvider"
)

# 准备输入
input_text = "人工智能的未来是"
input_ids = tokenizer.encode(input_text, return_tensors="np").astype(np.int64)

# 执行推理
outputs = ort_session.run(
    None,
    {"input_ids": input_ids}
)

# 处理输出
output_ids = outputs[0]
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
