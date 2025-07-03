from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BM25Retriever

# 创建向量检索器
vector_retriever = VectorIndexRetriever(index=vector_index)

# 创建BM25检索器
bm25_retriever = BM25Retriever.from_defaults(index=vector_index)

# 创建混合检索器
from llama_index.core.retrievers import HybridRetriever
hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    keyword_retriever=bm25_retriever,
    similarity_top_k=2,
)

# 创建查询引擎
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever
)