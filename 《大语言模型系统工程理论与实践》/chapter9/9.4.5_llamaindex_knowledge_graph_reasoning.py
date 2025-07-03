# 假设知识图谱已构建完成，命名为 kg_index
# 创建知识图谱查询引擎（具备推理能力）
query_engine = kg_index.as_query_engine()
# 执行具有推理性质的自然语言查询
response = query_engine.query("谁是量子力学发展过程中与爱因斯坦有争论的人？")
# 输出推理结果
print(response)