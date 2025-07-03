from llama_index.core.query_engine import SubQuestionQueryEngine

# 创建子问题查询引擎
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine=base_query_engine,
    verbose=True
)