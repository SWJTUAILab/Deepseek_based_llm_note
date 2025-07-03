from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# 使用本地嵌入模型
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 使用Ollama的gemma3:1b模型
llm = Ollama(
    model="gemma3:1b",
    request_timeout=120.0
)

# 加载本地文档数据
documents = SimpleDirectoryReader("data").load_data()

# 创建向量索引，指定使用本地嵌入模型
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# 构建查询引擎并执行问题检索，指定使用Ollama LLM
query_engine = index.as_query_engine(llm=llm, response_mode="compact")
response = query_engine.query("什么是知识图谱？")

# 打印结果
print("查询结果:")
print(response)