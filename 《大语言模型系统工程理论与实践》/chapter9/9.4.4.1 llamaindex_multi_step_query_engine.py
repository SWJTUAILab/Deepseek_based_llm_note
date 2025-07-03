from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# 加载文档和索引
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
# 创建多步骤推理引擎
multi_step_engine = MultiStepQueryEngine.from_defaults(query_engine=index.as_query_engine())
# 提出复杂问题（需要多步思考）
response = multi_step_engine.query("请先解释碳排放的定义，再说明其对气候变化的影响。")
print(response)