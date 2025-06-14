from llama_index.core import VectorStoreIndex

def create_vector_index(documents):
    """
    创建向量存储索引
    """
    index = VectorStoreIndex.from_documents(documents)
    return index

def create_basic_query_engine(index):
    """
    创建基本查询引擎
    """
    query_engine = index.as_query_engine()
    return query_engine
