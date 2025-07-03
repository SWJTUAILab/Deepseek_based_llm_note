from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# 加载文档并构建向量索引
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
# 构建基础查询引擎
basic_query_engine = index.as_query_engine()
# 发起查询
response = basic_query_engine.query("介绍一下碳中和的基本概念。")
print(response)