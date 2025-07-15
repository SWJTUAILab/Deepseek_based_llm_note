# 🚀 第九章代码验证报告

本项目包含《大语言模型系统工程理论与实践》第9章的学习代码，主要涵盖LlamaIndex框架的各种查询和检索技术。

## 📁 项目结构

```
chapter9/
├── 📂 data/                           # 示例数据文件
│   ├── artificial_intelligence.txt
│   ├── knowledge_graph_introduction.txt
│   └── large_language_models.txt
├── 📂 test_results/                   # 测试结果和截图
├── 9.4.2 llamaindex_vector_query.py
├── 9.4.3 llamaindex_auto_query_decomposition.py
├── 9.4.3 llamaindex_context_aware_retrieval.py
├── 9.4.3 llamaindex_hybrid_retrieval.py
├── 9.4.3 llamaindex_reranking_retrieval.py
├── 9.4.3 llamaindex_router_retrieval.py
├── 9.4.4.1 llamaindex_basic_query_engine.py
├── 9.4.4.1 llamaindex_knowledge_graph_query_engine.py
├── 9.4.4.1 llamaindex_multi_step_query_engine.py
├── 9.4.4.1 llamaindex_sql_query_engine.py
├── 9.4.4.1 llamaindex_summary_query_engine.py
├── 9.4.4.2 llamaindex_compact_synthesis.py
├── 9.4.4.2 llamaindex_refine_synthesis.py
├── 9.4.4.2 llamaindex_tree_synthesis.py
├── 9.4.5_llamaindex_hybrid_query.py
├── 9.4.5_llamaindex_knowledge_graph_construction.py
├── 9.4.5_llamaindex_knowledge_graph_query.py
├── 9.4.5_llamaindex_knowledge_graph_reasoning.py
├── requirements.txt
└── README.md
```

## ⚙️ 环境要求

- 🐍 Python 3.8+
- 🪟 Windows 10/11 (当前项目在Windows环境下开发)
- 💾 至少8GB内存
- 🚀 推荐使用GPU加速（可选）

## 🔧 安装步骤

### 1. 📥 克隆项目

```bash
git clone <项目地址>
cd chapter9
```

### 2. 🌍 创建虚拟环境

```bash
# 使用venv创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows PowerShell:
venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat

# Linux/Mac:
source venv/bin/activate
```

### 3. 📦 安装Python依赖

```bash
# 升级pip
python -m pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

### 4. 🤖 安装Ollama

#### Windows安装方法：

1. 访问 [Ollama官网](https://ollama.ai/download) 下载Windows版本
2. 运行安装程序并按照提示完成安装
3. 重启终端或PowerShell

#### 验证安装：

```bash
ollama --version
```

### 5. 📚 下载Gemma模型

```bash
# 下载Gemma-1b模型（推荐用于学习）
ollama pull gemma:1b

# 或者下载Gemma-4b模型甚至更大的模型（需要更多内存）
ollama pull gemma:4b

# 验证模型下载
ollama list
```

## 🎯 运行示例

### 基础向量查询示例

```bash
python "9.4.2 llamaindex_vector_query.py"
```

### 混合检索示例

```bash
python "9.4.3 llamaindex_hybrid_retrieval.py"
```

### 知识图谱查询示例

```bash
python "9.4.5_llamaindex_knowledge_graph_query.py"
```

## 🧩 主要功能模块

### 🔍 9.4.2 - 向量查询
- 基础的向量存储和检索功能
- 支持文档索引和相似度搜索

### 🔎 9.4.3 - 高级检索技术
- 自动查询分解
- 上下文感知检索
- 混合检索
- 重排序检索
- 路由检索

### ⚡ 9.4.4 - 查询引擎
- 基础查询引擎
- 知识图谱查询引擎
- 多步查询引擎
- SQL查询引擎
- 摘要查询引擎

### 🔄 9.4.4.2 - 合成策略
- 紧凑合成
- 细化合成
- 树状合成

### 🕸️ 9.4.5 - 知识图谱
- 知识图谱构建
- 知识图谱查询
- 知识图谱推理
- 混合查询

## ⚠️ 注意事项

1. **💾 内存使用**：运行较大的模型（如Gemma-7b）需要至少16GB内存
2. **⏳ 首次运行**：首次运行脚本时，LlamaIndex会自动下载必要的模型文件
3. **🌐 网络连接**：确保网络连接正常，用于下载模型和依赖
4. **📁 文件路径**：Windows系统下注意文件路径中的空格，建议使用引号包围文件名

## 🔧 故障排除

### 常见问题

1. **🔌 Ollama连接失败**
   ```bash
   # 检查Ollama服务状态
   ollama serve
   ```

2. **📥 模型下载失败**
   ```bash
   # 重新下载模型
   ollama pull gemma:2b
   ```

3. **📦 依赖安装失败**
   ```bash
   # 清理pip缓存
   pip cache purge
   # 重新安装
   pip install -r requirements.txt
   ```

4. **🔓 虚拟环境激活失败**
   ```bash
   # PowerShell执行策略问题
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

## 📖 学习建议

1. 📚 按照章节顺序逐步学习各个模块
2. 🎯 先运行基础示例，理解核心概念
3. 🔧 修改示例代码中的参数，观察不同效果
4. 📊 查看`test_results/`目录中的运行结果
5. 🧠 结合教材内容，深入理解每个技术的原理和应用场景

## 🤝 贡献

欢迎提交Issue和Pull Request
