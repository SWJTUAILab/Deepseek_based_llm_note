from data_connectors import load_documents
from vector_index import create_vector_index, create_basic_query_engine
from hybrid_retrieval import create_hybrid_retriever, create_hybrid_query_engine
from query_decomposition import create_contextual_query_engine
from knowledge_graph import create_knowledge_graph, create_hybrid_graph_query_engine

def main():
    # 加载文档
    documents = load_documents("data")
    
    # 创建向量索引
    vector_index = create_vector_index(documents)
    
    # 创建基本查询引擎
    basic_engine = create_basic_query_engine(vector_index)
    response = basic_engine.query("什么是知识图谱？")
    print("基本查询结果:", response)
    
    # 创建混合检索引擎
    hybrid_retriever = create_hybrid_retriever(vector_index)
    hybrid_engine = create_hybrid_query_engine(hybrid_retriever)
    response = hybrid_engine.query("解释RAG系统的工作原理")
    print("混合检索结果:", response)
    
    # 创建上下文感知查询引擎
    context_engine = create_contextual_query_engine(documents)
    response = context_engine.query("请总结一下这份政策的主要内容，并说明与上一版政策的关键变化有哪些？")
    print("上下文感知查询结果:", response)
    
    # 创建知识图谱和混合图查询引擎
    kg_index = create_knowledge_graph(documents)
    hybrid_graph_engine = create_hybrid_graph_query_engine(kg_index, vector_index)
    response = hybrid_graph_engine.query("谁是量子力学发展过程中与爱因斯坦有争论的人？")
    print("知识图谱查询结果:", response)

if __name__ == "__main__":
    main()
