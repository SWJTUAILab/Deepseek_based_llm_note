from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Step 1：加载文档
documents = SimpleDirectoryReader("data").load_data()
# Step 2：构建基础向量索引和查询引擎
index = VectorStoreIndex.from_documents(documents)
base_query_engine = index.as_query_engine()
# Step 3：构建支持上下文感知的子问题查询引擎
contextual_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine=base_query_engine,
    verbose=True  # 可选，打印中间子问题分解与回答过程
)
# Step 4：进行多步骤或上下文依赖的复杂查询
response = contextual_query_engine.query(
    "请总结一下这份政策的主要内容，并说明与上一版政策的关键变化有哪些？"
)
# Step 5：输出结果
print(response)