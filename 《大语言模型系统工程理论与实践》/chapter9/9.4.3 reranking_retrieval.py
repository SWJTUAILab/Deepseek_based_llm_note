from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

def create_reranking_query_engine(retriever, top_n=2):
    """
    创建带重排序功能的查询引擎
    """
    # 创建重排序处理器
    rerank = SentenceTransformerRerank(
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_n=top_n
    )
    
    # 创建查询引擎
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[rerank]
    )
    
    return query_engine
