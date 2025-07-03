from llama_index.core import VectorStoreIndex, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# 使用本地嵌入模型
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 使用本地Ollama LLM
llm = Ollama(
    model="gemma3:1b",
    request_timeout=120.0
)

# 1. 准备示例文档
documents = [
    Document(text="苹果是一种水果，富含维生素。"),
    Document(text="香蕉是黄色的热带水果，味道甜美。"),
    Document(text="计算机科学是研究计算机系统和算法的学科。"),
    Document(text="人工智能正在改变世界的各个领域。"),
]

# 2. 构建向量索引，指定使用本地嵌入模型
vector_index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# 3. 创建向量检索器
vector_retriever = VectorIndexRetriever(index=vector_index)

# 4. 创建相似度后处理器（替代BM25的混合检索效果）
similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

# 5. 创建查询引擎（使用向量检索 + 相似度过滤 + 本地LLM）
query_engine = RetrieverQueryEngine.from_args(
    retriever=vector_retriever,
    node_postprocessors=[similarity_postprocessor],
    llm=llm
)

# 6. 测试查询
queries = [
    "什么是水果？",
    "介绍人工智能",
    "计算机科学包含什么内容？",
    "香蕉的特点是什么？"
]

for q in queries:
    print(f"查询: {q}")
    response = query_engine.query(q)
    print(f"结果: {response}\n")
