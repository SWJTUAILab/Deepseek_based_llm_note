# 使用GPTQ进行INT4量化的示例代码
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# 加载模型和分词器
model_name = "The path where the model compression package is decompressed"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
output_dir="The path where you want to save"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,  # 对于 Llama 模型通常设为 False
    )
model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        trust_remote_code=True,
        device_map="auto",  # 自动选择设备
    ).to(device)  # 确保模型在正确设备上

# 准备校准数据
raw_texts = [
                    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
                    "大语言模型是一种基于深度学习的自然语言处理模型，它通过预训练和微调来理解和生成人类语言。",
                    "量化是一种模型压缩技术，通过降低模型参数的精度来减小模型大小并加速推理。"
                ] * 30  # 增大数据集确保足够token
calibration_dataset = []
for text in raw_texts:
    # 使用tokenizer编码文本
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,  # 限制长度防止内存问题
        return_tensors="pt"
    )
    # 移动到模型所在设备
    encoded = {k: v.to(device) for k, v in encoded.items()}
    calibration_dataset.append(encoded)
model = model.float()
model.quantize(calibration_dataset)
model.save_quantized(output_dir, use_safetensors=True)
tokenizer.save_pretrained(output_dir)
print(f"量化模型已部署到 {output_dir}")




# 加载量化后的模型进行推理
quantized_model = AutoModelForCausalLM.from_pretrained(
    "The path where you want to save",
    device_map="auto",  # 自动管理设备映射
    load_in_4bit=True
)

# 生成文本
input_text = "请介绍一下人工智能的发展历史"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
output_ids = quantized_model.generate(
    input_ids,
    max_length=2048,
    temperature=0.7,
    top_p=0.9
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
