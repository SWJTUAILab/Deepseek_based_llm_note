from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# 加载本地文档数据
documents = SimpleDirectoryReader("data").load_data()
# 创建向量索引
index = VectorStoreIndex.from_documents(documents)
# 构建查询引擎并执行问题检索
query_engine = index.as_query_engine()
response = query_engine.query("什么是知识图谱？")
