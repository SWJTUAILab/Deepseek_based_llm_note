# 创建支持 refine 模式的查询引擎
refine_engine = index.as_query_engine(response_mode="refine")
# 查询：逐步精炼多个节点形成最终回答
response = refine_engine.query("请总结这篇文章中关于数字化转型的观点。")
print(response)