# 创建支持 compact 模式的查询引擎
compact_engine = index.as_query_engine(response_mode="compact")
# 查询：适合处理大量内容的语义压缩总结
response = compact_engine.query("归纳这本书中关于人工智能发展的核心论点。")
print(response)