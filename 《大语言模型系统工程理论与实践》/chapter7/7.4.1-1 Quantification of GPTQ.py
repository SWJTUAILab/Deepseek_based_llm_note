import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def quantize_model_with_autogptq(
        model_name="The path where the model compression package is decompressed",
        output_dir="The path where you want to save",
        bits=4
):
    #确保使用CUDA设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    #加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #确保tokenizer有padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #创建量化配置
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,
        desc_act=False,  # 对于 Llama 模型通常设为 False
    )

    #加载模型到指定设备
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        trust_remote_code=True,
        device_map="auto",  # 自动选择设备
    ).to(device)  # 确保模型在正确设备上

    #准备校准数据集并移到设备上
    raw_texts = [
                    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
                    "大语言模型是一种基于深度学习的自然语言处理模型，它通过预训练和微调来理解和生成人类语言。",
                    "量化是一种模型压缩技术，通过降低模型参数的精度来减小模型大小并加速推理。"
                ] * 30  # 增大数据集确保足够token

    #准备校准数据集
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

    #执行量化
    model = model.float()
    model.quantize(calibration_dataset)

    #保存量化模型
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)

    print(f"量化模型已保存到 {output_dir}")
    return model, tokenizer


# 使用示例
if __name__ == "__main__":
    # 量化前清空缓存
    torch.cuda.empty_cache()

    quantized_model, tokenizer = quantize_model_with_autogptq()

    # 测试量化后的模型
    input_text = "人工智能的未来发展趋势是"

    # 准备输入并移到模型设备
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(quantized_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = quantized_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)