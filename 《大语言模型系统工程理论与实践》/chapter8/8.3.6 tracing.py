from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback

# 初始化模型（确保你已经配置了 OpenAI API 密钥）
llm = ChatOpenAI(model="gpt-4o-mini")

# 提示模板，鼓励思维链（CoT）推理
template = """
你是一位逻辑严密的数学专家。
请逐步思考并解决以下问题：
{question}
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# 构建链
cot_chain = LLMChain(llm=llm, prompt=prompt)

# 使用回调统计 token 数量和费用
with get_openai_callback() as cb:
    result = cot_chain.run(question="如果一个列车以每小时80公里的速度行驶4小时，它行驶了多少公里？")

    # 输出结果
    print("模型输出：")
    print(result)

    # 输出统计信息
    print("\n执行统计：")
    print(f"总令牌数: {cb.total_tokens}")
    print(f"提示令牌: {cb.prompt_tokens}")
    print(f"完成令牌: {cb.completion_tokens}")
    print(f"成功的请求: {cb.successful_requests}")
    print(f"总花费: ${cb.total_cost:.6f}")
