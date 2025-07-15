# 创建知识图谱查询引擎
query_engine = kg_index.as_query_engine()
# 执行查询
response = query_engine.query("爱因斯坦的主要贡献是什么？")
