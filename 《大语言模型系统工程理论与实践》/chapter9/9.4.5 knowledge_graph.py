from llama_index.core import KnowledgeGraphIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.query_engine import ComposableGraphQueryEngine

def create_knowledge_graph(documents, max_triplets_per_chunk=10):
    """
    从文档创建知识图谱索引
    """
    # 创建图存储
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    
    # 创建知识图谱索引
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=max_triplets_per_chunk
    )
    
    return kg_index

def create_hybrid_graph_query_engine(kg_index, vector_index, verbose=True):
    """
    创建混合查询引擎，结合知识图谱和向量索引
    """
    query_engine = ComposableGraphQueryEngine(
        kg_index=kg_index,
        vector_index=vector_index,
        verbose=verbose
    )
    
    return query_engine
