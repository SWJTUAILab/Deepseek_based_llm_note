from llama_index.core.postprocessor import SentenceTransformerRerank

# 创建重排序处理器
rerank = SentenceTransformerRerank(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_n=2
)

# 创建查询引擎
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    node_postprocessors=[rerank]
)