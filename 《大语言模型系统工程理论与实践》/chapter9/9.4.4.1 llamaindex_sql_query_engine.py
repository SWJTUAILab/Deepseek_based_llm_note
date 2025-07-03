from llama_index.core.query_engine import SQLQueryEngine
from sqlalchemy import create_engine
from llama_index.core.indices.struct_store import SQLStructStoreIndex
# 创建数据库连接（使用 SQLite 示例）
engine = create_engine("sqlite:///example.db")
# 构建 SQL 索引
sql_index = SQLStructStoreIndex.from_sql_database(sql_database=engine)
# 构建 SQL 查询引擎
sql_query_engine = SQLQueryEngine(sql_index)
# 发起结构化自然语言查询
response = sql_query_engine.query("有哪些员工的工资高于 10000 元？")
print(response)