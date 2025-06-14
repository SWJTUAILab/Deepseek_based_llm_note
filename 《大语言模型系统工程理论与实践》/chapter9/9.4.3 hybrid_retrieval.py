from llama_index.core.retrievers import VectorIndexRetriever, BM25Retriever, HybridRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

def create_hybrid_retriever(vector_index):
    """
    创建混合检索器，结合向量检索和BM25检索
    """
    # 创建向量检索器
    vector_retriever = VectorIndexRetriever(index=vector_index)
    
    # 创建BM25检索器
    bm25_retriever = BM25Retriever.from_defaults(index=vector_index)
    
    # 创建混合检索器
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=bm25_retriever,
        similarity_top_k=2,
    )
    
    return hybrid_retriever

def create_hybrid_query_engine(hybrid_retriever):
    """
    基于混合检索器创建查询引擎
    """
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever
    )
    return query_engine
