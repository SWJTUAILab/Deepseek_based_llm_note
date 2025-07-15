import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM
import os

# Specify paths and hyperparameters for quantization
model_path = 'The path where the model compression package is decompressed'
quant_path = "The path where you want to save"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
# Load your tokenizer and model with AutoAWQ
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoAWQForCausalLM.from_pretrained( model_path,
    safetensors=True,
    device_map='cuda',  # 不使用 auto 分配
    low_cpu_mem_usage=True,
    torch_dtype="auto",  # 可以尝试 "auto" 或 "torch.float16"
    local_files_only=True)
# model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=True, device_map={"": "cpu"}, local_files_only=True,low_cpu_mem_usage=True)
calib_data = [
    # 人工智能基础
    "深度学习模型需要大量计算资源和训练数据支持运作",
    "神经网络模仿人脑结构实现复杂的认知功能",
    "机器学习算法分析数据建立预测模型方法",
    "人工智能技术正改变各行各业运作方式现状",
    # 自然语言处理
    "语言模型生成流畅文本具备对话问答能力",
    "情感分析识别用户情绪用于评论监测",
    "机器翻译技术跨越语言障碍促进交流",
    "文本摘要提取关键信息提高阅读效率",
    # 计算机视觉
    "图像识别技术能够准确分类物体场景",
    "目标检测定位物品位置应用于自动驾驶",
    "人脸识别系统实现身份验证安防监控",
    "图像分割区分不同区域医学图像应用",
    # 伦理与治理
    "算法偏见可能导致歧视需伦理审查",
    "数据隐私保护应当符合规范防止泄露",
    "人机协作模式将成为未来主流趋势",
    "监管框架确保技术安全可控发展",
    # 行业应用
    "智能制造机器人提升生产效率质量",
    "金融风控系统检测异常交易行为",
    "智慧医疗辅助诊断提高准确效率",
    "智能交通优化路线减少拥堵延误",
    # 技术发展
    "大模型技术突破语言理解生成能力",
    "向量嵌入表示词语语义关系信息",
    "强化学习训练决策智能体玩游戏",
    "生成模型创造新颖内容图像文本",
    # 研究前沿
    "具身智能实体世界交互学习发展",
    "多模态融合视觉语言共同理解",
    "可解释性方法揭示模型决策过程",
    "持续学习适应环境动态变化",
    # 未来发展
    "脑机接口连接人脑计算机交互",
    "量子计算加速训练突破限制",
    "通用人工智能长远发展目标",
    "人机共生社会需要伦理指引"
]
model.quantize(tokenizer=tokenizer, quant_config=quant_config,calib_data=calib_data)
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
tokenizer.save_pretrained(quant_path)
print(f"量化模型已保存到 {quant_path}")
# 载入量化后的模型
quant_path = "The path where you want to save"
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_quantized(quant_path, device_map="auto")
# 输入一句话
input_text = "人工智能的未来发展趋势是"
# 编码输入
inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
# 生成
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
# 解码输出
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("模型生成结果：\n", result)
