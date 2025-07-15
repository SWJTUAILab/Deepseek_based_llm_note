#### 9.4.2 向量检索相关

##### 9.4.2 llamaindex_vector_query.py
1. **功能**: 基于向量存储的文档检索和问答系统
2. **依赖**: llama-index-core, llama-index-embeddings-huggingface, llama-index-llms-ollama, sentence-transformers
3. **运行结果**:
   - 成功加载本地文档数据
   - 使用sentence-transformers/all-MiniLM-L6-v2进行向量化
   - 通过Ollama的gemma3:1b模型生成回答
   - 输出关于"什么是知识图谱？"的详细回答
4. **运行**: `python "9.4.2 llamaindex_vector_query.py"`

#### 9.4.3 高级检索策略

##### 9.4.3 llamaindex_auto_query_decomposition.py
1. **功能**: 自动查询分解，将复杂问题拆分为子问题
2. **依赖**: llama-index-core
3. **运行结果**:
   - 需要先定义base_query_engine
   - 创建SubQuestionQueryEngine实例
   - 能够自动分解复杂查询为多个子问题
4. **注意**: 此文件为部分实现，需要配合其他查询引擎使用

##### 9.4.3 llamaindex_context_aware_retrieval.py
1. **功能**: 上下文感知的检索系统
2. **依赖**: llama-index-core
3. **运行结果**:
   - 加载文档并构建向量索引
   - 创建支持上下文感知的子问题查询引擎
   - 能够处理多步骤或上下文依赖的复杂查询
   - 输出政策总结和变化分析
4. **运行**: `python "9.4.3 llamaindex_context_aware_retrieval.py"`

##### 9.4.3 llamaindex_hybrid_retrieval.py
1. **功能**: 混合检索策略（向量检索 + 相似度过滤）
2. **依赖**: llama-index-core, llama-index-embeddings-huggingface, llama-index-llms-ollama
3. **运行结果**:
   - 创建示例文档并构建向量索引
   - 使用相似度后处理器过滤结果
   - 对多个测试查询返回相关回答
   - 输出水果、AI、计算机科学等相关信息
4. **运行**: `python "9.4.3 llamaindex_hybrid_retrieval.py"`

##### 9.4.3 llamaindex_reranking_retrieval.py
1. **功能**: 重排序检索，使用交叉编码器提升检索质量
2. **依赖**: llama-index-core, llama-index-embeddings-huggingface, llama-index-llms-ollama
3. **运行结果**:
   - 构建向量索引并创建检索器
   - 使用SentenceTransformerRerank进行结果重排序
   - 对测试查询返回高质量排序结果
   - 显示查询结果和来源节点数量
4. **运行**: `python "9.4.3 llamaindex_reranking_retrieval.py"`

##### 9.4.3 llamaindex_router_retrieval.py
1. **功能**: 路由检索，根据查询类型选择不同的查询引擎
2. **依赖**: llama-index-core
3. **运行结果**:
   - 创建RouterQueryEngine实例
   - 根据查询内容自动路由到相应的专业查询引擎
   - 需要预先定义science_engine、history_engine、art_engine
4. **注意**: 此文件为部分实现，需要配合多个专业查询引擎使用

#### 9.4.4 查询引擎类型

##### 9.4.4.1 llamaindex_basic_query_engine.py
1. **功能**: 基础查询引擎实现
2. **依赖**: llama-index-core
3. **运行结果**:
   - 加载文档并构建向量索引
   - 创建基础查询引擎
   - 输出关于碳中和概念的详细回答
4. **运行**: `python "9.4.4.1 llamaindex_basic_query_engine.py"`

##### 9.4.4.1 llamaindex_knowledge_graph_query_engine.py
1. **功能**: 知识图谱查询引擎
2. **依赖**: llama-index-core
3. **运行结果**:
   - 加载文档并构建知识图谱索引
   - 创建知识图谱查询引擎
   - 输出关于爱因斯坦相对论的实体关系推理结果
4. **运行**: `python "9.4.4.1 llamaindex_knowledge_graph_query_engine.py"`

##### 9.4.4.1 llamaindex_multi_step_query_engine.py
1. **功能**: 多步骤推理查询引擎
2. **依赖**: llama-index-core
3. **运行结果**:
   - 创建多步骤推理引擎
   - 能够处理需要分步思考的复杂问题
   - 输出碳排放定义和气候变化影响的详细分析
4. **运行**: `python "9.4.4.1 llamaindex_multi_step_query_engine.py"`

##### 9.4.4.1 llamaindex_sql_query_engine.py
1. **功能**: SQL查询引擎，支持自然语言到SQL的转换
2. **依赖**: llama-index-core, sqlalchemy
3. **运行结果**:
   - 连接SQLite数据库
   - 构建SQL结构存储索引
   - 将自然语言查询转换为SQL并执行
   - 输出符合条件的员工信息
4. **运行**: `python "9.4.4.1 llamaindex_sql_query_engine.py"`

##### 9.4.4.1 llamaindex_summary_query_engine.py
1. **功能**: 摘要型查询引擎
2. **依赖**: llama-index-core
3. **运行结果**:
   - 创建摘要型查询引擎
   - 对技术报告进行总结
   - 输出文档的主要观点摘要
4. **运行**: `python "9.4.4.1 llamaindex_summary_query_engine.py"`

##### 9.4.4.2 合成策略

###### 9.4.4.2 llamaindex_compact_synthesis.py
1. **功能**: Compact模式合成，适合处理大量内容的语义压缩
2. **依赖**: llama-index-core
3. **运行结果**:
   - 创建支持compact模式的查询引擎
   - 对大量内容进行语义压缩总结
   - 输出关于AI发展的核心论点归纳
4. **注意**: 需要预先定义index变量

###### 9.4.4.2 llamaindex_refine_synthesis.py
1. **功能**: Refine模式合成，逐步精炼多个节点形成最终回答
2. **依赖**: llama-index-core
3. **运行结果**:
   - 创建支持refine模式的查询引擎
   - 逐步精炼多个文档节点
   - 输出关于数字化转型观点的总结
4. **注意**: 需要预先定义index变量

###### 9.4.4.2 llamaindex_tree_synthesis.py
1. **功能**: 树结构合成，根据结构分层总结
2. **依赖**: llama-index-core
3. **运行结果**:
   - 创建树结构索引
   - 使用树式合成查询引擎
   - 按照章节结构总结技术白皮书
4. **注意**: 需要预先定义documents变量

#### 9.4.5 知识图谱与混合查询

##### 9.4.5_llamaindex_hybrid_query.py
1. **功能**: 混合查询引擎，结合知识图谱和向量检索
2. **依赖**: llama-index-core
3. **运行结果**:
   - 创建ComposableGraphQueryEngine
   - 结合知识图谱和向量索引进行查询
   - 输出混合检索结果
4. **注意**: 需要预先定义kg_index和vector_index变量

##### 9.4.5_llamaindex_knowledge_graph_reasoning.py
1. **功能**: 知识图谱推理查询
2. **依赖**: llama-index-core
3. **运行结果**:
   - 创建具备推理能力的知识图谱查询引擎
   - 执行具有推理性质的自然语言查询
   - 输出关于量子力学发展过程中与爱因斯坦争论的人的推理结果
4. **注意**: 需要预先定义kg_index变量

##### 9.4.5_llamaindex_knowledge_graph_construction.py
1. **功能**: 知识图谱构建
2. **依赖**: llama-index-core
3. **运行结果**:
   - 创建图存储和存储上下文
   - 从文档构建知识图谱索引
   - 设置每个块的最大三元组数量
4. **注意**: 需要预先定义documents变量

##### 9.4.5_llamaindex_knowledge_graph_query.py
1. **功能**: 知识图谱查询
2. **依赖**: llama-index-core
3. **运行结果**:
   - 创建知识图谱查询引擎
   - 执行关于爱因斯坦主要贡献的查询
   - 输出基于知识图谱的实体关系回答
4. **注意**: 需要预先定义kg_index变量

#### 总体评估

##### 完整性分析
- **完整实现**: 9.4.2, 9.4.3 (context_aware, hybrid, reranking), 9.4.4.1 (basic, knowledge_graph, multi_step, sql, summary)
- **部分实现**: 9.4.3 (auto_query_decomposition, router), 9.4.4.2 (所有文件), 9.4.5 (所有文件)

##### 依赖关系
- 所有文件都需要llama-index-core
- 部分文件需要额外的嵌入模型和LLM支持
- SQL查询引擎需要sqlalchemy
- 重排序功能需要sentence-transformers

##### 运行环境
- Python 3.8+
- 本地Ollama服务运行gemma3:1b模型
- 足够的计算资源支持向量化和推理
- 网络连接用于下载预训练模型（首次运行）