# 第二章代码验证

## 系统操作环境

  * **操作系统** ：Windows 11
  * **CPU** ：Intel(R) Core(TM) i9-14900

## 依赖安装

  1. 安装 transformers、torch：`pip install transformers torch`
  2. 安装 numpy：`pip install numpy`
  3. 安装 gensim：`pip install gensim`
  4. 安装 allennlp：`pip install allennlp`

## 代码及运行结果解释

### 2.1.1 分词器示例

  * **安装** ：transformers 库，NumPy 库。python3.12 可用。
  * **示例运行结果** ：将文本经过分词器拆分编码后形成数字 ID,再将 ID 解码成为文本，过程可逆。

### 2.2.2 静态词嵌入

  * **安装** ：gensim 库。python3.12 可用。
  * **示例运行结果** ：将输入的文本映射到低维稠密向量空间中。

### 2.2.3 ELMo

  * **安装** ：allennlp 库，pytorch 库。要下载 python3.7！
  * **示例运行结果** ：各维度含义：2：批次大小（batch_size）→ 输入了 2 条句子。7：序列长度（sequence_length）→ 两条句子的最大词数（第一条 5 词，第二条 7 词，填充到 7）。1024：嵌入维度（embedding_dim）→ ELMo 预训练模型的输出维度（由 options_file 定义，通常为 1024）。

### 2.2.3 GPT

  * **安装** ：transformers 库，pytorch 库。python3.12 可用。
  * **示例运行结果** ：代码中打印 last_hidden_states 的形状，并展示其维度信息。输出中的 [1,6,768] 表示维度，其中 1 是批次大小（这里处理单条文本，所以批次为 1）；6 是序列长度（文本经分词、填充 / 截断后得到 6 个词元）；768 是隐藏状态维度（GPT-2 模型输出的每个词元隐藏状态的特征维度），说明成功获取到了模型对输入文本处理后的特征表示。

### 2.2.3 BERT

  * **安装** ：transformers 库，pytorch 库。python3.12 可用。
  * **示例运行结果** ：结果解释同上。

### 2.3.1-1 位置编码示例

  * **安装** ：NumPy 库。python3.12 可用。
  * **示例运行结果** ：代码的输出 Shape after adding positional encoding: torch.Size([32, 60, 512]) 表示：32：批次大小（batch_size）→ 一次处理 32 条文本。60：序列长度（seq_len）→ 每条文本有 60 个词（或词元）。512：特征维度（d_model）→ 每个词的嵌入维度是 512，添加位置编码后维度不变（逐元素相加，不改变形状）。这表明位置编码成功添加到输入嵌入中，只有值被修改。

### 2.3.1-2 可学习位置编码

  * **安装** ：pytorch 库。python3.12 可用。
  * **示例运行结果** ：结果解释同上。

### 2.3.2-3 T5 位置编码

  * **安装** ：pytorch 库。python3.12 可用。
  * **示例运行结果** ：在代码末尾输入如下，验证代码。结果输出：0,1,2：相对位置较小（如 0,1,2 或 -1 经处理后），映射到精确桶（前半部分桶，直接对应距离）。31：相对位置 -1 超出 “精确桶” 范围（如 num_buckets=32 时，max_exact=16），映射到近似桶（通过对数距离或其他策略分桶）。

### 2.3.2-5 Alibi 位置偏置

  * **安装** ：pytorch 库。python3.12 可用。
  * **示例运行结果** ：在代码的末尾加上 `print(attention_scores_with_alibi.shape)` 打印语句，输出：输出对应 Alibi 偏置矩阵的形状，其中：1：批次大小（Batch Size）→ 示例中模拟单条样本（batch=1）。8：注意力头数量（Num Heads）→ 代码中 num_heads=8，每个头对应一组独立的偏置。50：查询序列长度（Query Length, q_len）→ 代码中 q_len=50，表示 Query 序列的元素数量。60：键序列长度（Key Length, k_len）→ 代码中 k_len=60，表示 Key 序列的元素数量。

### 2.3.3-5 旋转位置编码

  * **安装** ：pytorch 库。python3.12 可用。
  * **示例运行结果** ：输出的 `torch.Size([1, 10, 8, 64])` 表示：1：批次大小（Batch）→ 一次处理 1 条文本。10：序列长度（SeqLen）→ 文本有 10 个词（或词元）。8：头数（NumHeads）→ 多 - head 注意力的头数量（RoPE 对每个头独立处理）。64：头维度（HeadDim）→ 每个头的特征维度（RoPE 作用于该维度，通过复数旋转编码位置）。

## 问题记录

  1. **numpy 与 python 版本对应** ：安装 allennlp 时，python、pytorch 版本过高会导致 allennlp 中的 spacy 库不兼容，从而报错。这里选用 python3.7，安装这个库，可以正常运行。
  2. **GPT 代码警告** ：提示 `invalid escape sequence '\g'`，是因为在 Python 字符串中，\ 是转义字符，路径里的 D:\gpt2 若想表示原始路径，应该用 r"D:\gpt2"（原始字符串，\ 不被视为转义）或者 D:\\gpt2（转义 \），不过这一般不影响代码实际功能运行，只是语法规范提醒。
