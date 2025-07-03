from llama_index.core.query_engine import ComposableGraphQueryEngine

# 创建混合查询引擎
query_engine = ComposableGraphQueryEngine(
    kg_index=kg_index,
    vector_index=vector_index,
    verbose=True
)