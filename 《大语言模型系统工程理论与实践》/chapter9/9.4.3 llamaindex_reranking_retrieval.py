from llama_index.core import VectorStoreIndex, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# 1. 准备测试文档
documents = [
    Document(text="苹果是一种水果，富含维生素C和纤维，有助于消化。"),
    Document(text="香蕉是黄色的热带水果，味道甜美，富含钾元素。"),
    Document(text="计算机科学是研究计算机系统和算法的学科，包括数据结构、算法设计等。"),
    Document(text="人工智能正在改变世界的各个领域，包括医疗、教育、交通等。"),
    Document(text="机器学习是人工智能的一个分支，通过数据训练模型。"),
    Document(text="深度学习使用神经网络来模拟人脑的学习过程。"),
]

# 2. 构建向量索引
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# 3. 创建检索器
retriever = VectorIndexRetriever(index=vector_index)

# 4. 创建重排序处理器
rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_n=2
)

# 使用本地Ollama LLM
llm = Ollama(
    model="gemma3:1b",
    request_timeout=120.0
)

# 5. 创建查询引擎
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    node_postprocessors=[rerank],
    llm=llm
)

# 6. 测试查询
test_queries = [
    "什么是水果？",
    "介绍人工智能技术",
    "机器学习的基本概念",
    "计算机科学的研究内容"
]

print("=== 重排序检索测试 ===\n")

for query in test_queries:
    print(f"🔍 查询: {query}")
    response = query_engine.query(query)
    print(f"📝 回答: {response}")
    print(f"📊 来源节点数: {len(response.source_nodes)}")
    print("-" * 50)