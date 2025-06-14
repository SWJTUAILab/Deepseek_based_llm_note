def create_refine_engine(index):
    """
    创建重构式合成查询引擎
    """
    refine_engine = index.as_query_engine(response_mode="refine")
    return refine_engine

def create_compact_engine(index):
    """
    创建压缩式合成查询引擎
    """
    compact_engine = index.as_query_engine(response_mode="compact")
    return compact_engine

def create_tree_synthesis_engine(documents):
    """
    创建树式合成查询引擎
    """
    from llama_index.core.indices.tree import TreeIndex
    # 使用树结构索引
    tree_index = TreeIndex.from_documents(documents)
    # 创建树式合成查询引擎
    tree_engine = tree_index.as_query_engine(response_mode="tree_summarize")
    return tree_engine
