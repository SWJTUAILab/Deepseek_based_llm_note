from llama_index.core.query_engine import SummaryQueryEngine
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# 加载文档和索引
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
# 构建摘要型查询引擎
summary_engine = SummaryQueryEngine.from_args(index)
# 提出总结类问题
response = summary_engine.query("请总结这份技术报告的主要观点。")
print(response)