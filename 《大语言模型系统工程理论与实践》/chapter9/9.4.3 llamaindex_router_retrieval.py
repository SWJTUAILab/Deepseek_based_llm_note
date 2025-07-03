from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine import RouterQueryEngine

# 创建路由查询引擎
query_engine = RouterQueryEngine.from_defaults(
    selector=LLMSingleSelector.from_defaults(),
    query_engines={
        "科学": science_engine,
        "历史": history_engine,
        "艺术": art_engine
    },
    verbose=True
)