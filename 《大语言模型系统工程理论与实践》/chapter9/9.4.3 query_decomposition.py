from llama_index.core.query_engine import SubQuestionQueryEngine

def create_subquestion_engine(base_query_engine, verbose=True):
    """
    创建子问题查询引擎，用于自动查询分解
    """
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine=base_query_engine,
        verbose=verbose
    )
    return query_engine

def create_contextual_query_engine(documents):
    """
    创建支持上下文感知的查询引擎
    """
    from llama_index.core import VectorStoreIndex
    
    # 构建基础向量索引和查询引擎
    index = VectorStoreIndex.from_documents(documents)
    base_query_engine = index.as_query_engine()
    
    # 构建支持上下文感知的子问题查询引擎
    contextual_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine=base_query_engine,
        verbose=True  # 可选，打印中间子问题分解与回答过程
    )
    
    return contextual_query_engine
