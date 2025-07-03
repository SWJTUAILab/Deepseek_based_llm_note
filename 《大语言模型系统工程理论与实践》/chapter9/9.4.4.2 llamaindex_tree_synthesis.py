from llama_index.core.indices.tree import TreeIndex
# 使用树结构索引
tree_index = TreeIndex.from_documents(documents)
# 创建树式合成查询引擎
tree_engine = tree_index.as_query_engine(response_mode="tree_summarize")
# 查询：根据结构分层总结
response = tree_engine.query("请按照章节结构总结这篇技术白皮书。")
print(response)