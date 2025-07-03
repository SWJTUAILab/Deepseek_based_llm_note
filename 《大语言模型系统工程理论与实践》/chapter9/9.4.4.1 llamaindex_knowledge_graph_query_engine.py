from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.core import SimpleDirectoryReader
# 加载文档
documents = SimpleDirectoryReader("data").load_data()
# 构建知识图谱索引
kg_index = KnowledgeGraphIndex.from_documents(documents)
# 创建知识图谱查询引擎
kg_query_engine = KnowledgeGraphQueryEngine(kg_index)
# 进行实体关系推理查询
response = kg_query_engine.query("哪位科学家提出了相对论？")
print(response)