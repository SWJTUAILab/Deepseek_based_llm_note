from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine import RouterQueryEngine

def create_router_query_engine(query_engines_dict, verbose=True):
    """
    创建路由查询引擎，根据查询内容选择合适的查询引擎
    """
    query_engine = RouterQueryEngine.from_defaults(
        selector=LLMSingleSelector.from_defaults(),
        query_engines=query_engines_dict,
        verbose=verbose
    )
    return query_engine
