import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch.nn.functional as F


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



def quantize_and_distill(
        teacher_model_name="The path where the model compression package is decompressed——large",
        student_model_name="The path where the model compression package is decompressed——small",
        output_dir="The path where you want to save",
        train_dataset=None,
        bits=4
):
    """
    结合量化和蒸馏优化大语言模型

    参数:
    - teacher_model_name: 教师模型名称或路径
    - student_model_name: 学生模型名称或路径
    - output_dir: 优化后模型的保存路径
    - train_dataset: 训练数据集
    - bits: 量化位数
    """
    # 第一步：量化教师模型以减少蒸馏过程中的资源消耗
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)

    # 创建量化器
    quantizer = GPTQQuantizer(
        bits=8,  # 教师模型使用较高精度的量化
        dataset=["量化校准文本示例"],
        tokenizer=tokenizer
    )

    # 量化教师模型
    quantized_teacher = quantizer.quantize_model(teacher_model)

    # 第二步：使用量化后的教师模型蒸馏学生模型
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name)

    # 执行蒸馏（使用前面定义的DistillationTrainer）
    distilled_student = distill_llm(
        teacher_model=quantized_teacher,
        student_model=student_model,
        train_dataset=train_dataset,
        output_dir=f"{output_dir}-temp"
    )

    # 第三步：量化蒸馏后的学生模型
    student_quantizer = GPTQQuantizer(
        bits=bits,
        dataset=["量化校准文本示例"],
        tokenizer=tokenizer
    )

    # 量化学生模型
    final_model = student_quantizer.quantize_model(distilled_student)

    # 保存最终模型
    final_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"量化蒸馏模型已保存到 {output_dir}")

    return final_model, tokenizer
