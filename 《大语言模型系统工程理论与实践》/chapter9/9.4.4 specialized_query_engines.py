from llama_index.core.query_engine import SummaryQueryEngine, MultiStepQueryEngine
from llama_index.core.indices.struct_store import SQLStructStoreIndex
from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.core.query_engine import SQLQueryEngine, KnowledgeGraphQueryEngine

def create_summary_engine(index):
    """
    创建总结查询引擎
    """
    summary_engine = SummaryQueryEngine.from_args(index)
    return summary_engine

def create_sql_query_engine(db_connection):
    """
    创建SQL查询引擎
    """
    # 构建SQL索引
    sql_index = SQLStructStoreIndex.from_sql_database(sql_database=db_connection)
    # 构建SQL查询引擎
    sql_query_engine = SQLQueryEngine(sql_index)
    return sql_query_engine

def create_kg_query_engine(documents):
    """
    创建知识图谱查询引擎
    """
    # 构建知识图谱索引
    kg_index = KnowledgeGraphIndex.from_documents(documents)
    # 创建知识图谱查询引擎
    kg_query_engine = KnowledgeGraphQueryEngine(kg_index)
    return kg_query_engine

def create_multi_step_engine(index):
    """
    创建多步查询引擎
    """
    multi_step_engine = MultiStepQueryEngine.from_defaults(
        query_engine=index.as_query_engine()
    )
    return multi_step_engine
