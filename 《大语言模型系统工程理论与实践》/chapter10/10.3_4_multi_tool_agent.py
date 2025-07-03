from langchain.agents import AgentType, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import tool

@tool("search_web", description="用于从网络搜索信息")
def search_web(query: str) -> str:
    return f"搜索结果：关于 {query} 的信息..."

@tool("calculator", description="用于执行数学表达式计算")
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{str(e)}"

llm = ChatOpenAI(temperature=0)
tools = [search_web, calculator]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

response = agent.invoke({"input": "法国的人口是多少？另外，23 * 45 是多少？"})
