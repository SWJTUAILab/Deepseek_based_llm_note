import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature

        if self.teacher_model is not None:
            self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 确保 inputs 和模型在同一设备
        device = model.device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        # 前向传播（学生模型）
        outputs = model(**inputs)
        student_logits = outputs.logits

        # 标签
        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError("Missing 'labels' in inputs for loss computation.")

        # 交叉熵损失（硬标签）
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # 没有教师模型就返回硬损失
        if self.teacher_model is None:
            return (hard_loss, outputs) if return_outputs else hard_loss

        # 教师模型前向传播（保持教师模型在其设备上）
        with torch.no_grad():
            teacher_device = self.teacher_model.device
            teacher_inputs = {k: v.to(teacher_device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits.to(device)  # 把教师输出移回学生模型所在设备

        # 蒸馏损失（软标签 KL 散度）
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(
            soft_student.view(-1, soft_student.size(-1)),
            soft_teacher.view(-1, soft_teacher.size(-1)),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        # 总损失
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return (loss, outputs) if return_outputs else loss


def distill_llm(
        teacher_model_name="The path where the model compression package is decompressed——large",
        student_model_name="The path where the model compression package is decompressed——small",  # 假设存在这样的小模型
        output_dir="The path where you want to save",
        train_dataset=None,
        alpha=0.5,
        temperature=2.0,
        batch_size=2,
        learning_rate=5e-5,
        num_epochs=1
):
    """
    使用知识蒸馏训练小型语言模型

    参数:
    - teacher_model_name: 教师模型名称或路径
    - student_model_name: 学生模型名称或路径
    - output_dir: 蒸馏后模型的保存路径
    - train_dataset: 训练数据集
    - alpha: 硬目标损失的权重
    - temperature: 软目标的温度参数
    - batch_size: 批大小
    - learning_rate: 学习率
    - num_epochs: 训练轮数
    """
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    if tokenizer.pad_token is None:
        # 优先使用eos_token作为pad token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"设置pad_token为eos_token: {tokenizer.pad_token}")
        else:
            # 如果eos_token也不存在，添加新的pad token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("添加了新的pad_token: [PAD]")
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)
    print(f"模型已加载到设备: {device}")
    # 如果添加了新的pad token，需要调整模型嵌入层大小
    if tokenizer.pad_token == '[PAD]':
        teacher_model.resize_token_embeddings(len(tokenizer))
        student_model.resize_token_embeddings(len(tokenizer))
        print("调整了模型嵌入层大小以适配新的pad token")
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        evaluation_strategy="no",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,  # 减少日志步数以更频繁地查看进度
        load_best_model_at_end=False,
        fp16=True,
        gradient_accumulation_steps=4,
        report_to="none"  # 禁用所有报告器以减少开销
    )

    # 创建蒸馏训练器
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        alpha=alpha,
        temperature=temperature,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        # 修复tokenizer弃用警告
        data_collator=None,  # 自动使用默认数据整理器
        tokenizer=tokenizer,  # 保留以修复弃用警告
    )

    # 训练学生模型
    print("开始知识蒸馏训练...")
    trainer.train()

    # 保存蒸馏后的模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"蒸馏模型已保存到 {output_dir}")

    return student_model, tokenizer


# 使用示例
if __name__ == "__main__":
    # 创建一个简单的训练数据集
    # 实际应用中应该使用更大的数据集
    from datasets import Dataset

    train_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "大语言模型是一种基于深度学习的自然语言处理模型，它通过预训练和微调来理解和生成人类语言。",
        "知识蒸馏是一种模型压缩技术，通过让小模型学习大模型的输出分布来提高小模型的性能。"
    ]


    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens


    train_dataset = Dataset.from_dict({"text": train_texts})
    tokenizer = AutoTokenizer.from_pretrained("E:\\Python_project\\SpotGeo_v2\\llama_model")
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    # 执行蒸馏
    distilled_model, tokenizer = distill_llm(train_dataset=train_dataset)

    # 测试蒸馏后的模型
    input_text = "人工智能的未来发展趋势是"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        output_ids = distilled_model.generate(
            input_ids,
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)
