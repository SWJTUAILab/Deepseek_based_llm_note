import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

# 加载模型和分词器
model_name = "The path where the model compression package is decompressed"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建模型实例
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # 使用FP16精度
)

# 正确的DeepSpeed推理配置（简化版本）
ds_config = {
    "dtype": torch.float16,   # 指定推理精度
    "mp_size": 2,             # 模型并行，使用2个GPU
    "replace_with_kernel_inject": True,
    "enable_cuda_graph": False  # Windows下建议关闭
}

# 初始化DeepSpeed推理引擎
model = deepspeed.init_inference(
    model,
    config=ds_config
)

# 生成文本
input_text = "请介绍一下人工智能的发展历史"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
output_ids = model.generate(
    input_ids,
    max_length=2048,
    temperature=0.7,
    top_p=0.9
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)