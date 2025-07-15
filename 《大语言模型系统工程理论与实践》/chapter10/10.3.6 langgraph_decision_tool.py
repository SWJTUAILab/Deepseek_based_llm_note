from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Union
import json

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, FunctionMessage]]
    current_tool: str
    tool_input: str
    tool_output: str
    next_step: str

def search_tool(query: str) -> str:
    return f"搜索结果：关于 {query} 的信息..."

def calculator_tool(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{str(e)}"

tools = {
    "search": search_tool,
    "calculator": calculator_tool
}

def agent_decision(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4-turbo")
    system_message = """你是一个智能助手，可以使用以下工具：
- search: 搜索网络获取信息
- calculator: 执行数学计算
分析用户意图，决定是否调用工具，或直接给出答案。"""
    response = llm.invoke([{"role": "system", "content": system_message}] + state["messages"])
    try:
        decision = json.loads(response.content)
        if decision["next_step"] == "use_tool":
            return {
                "messages": state["messages"],
                "current_tool": decision["tool"],
                "tool_input": decision["tool_input"],
                "next_step": "use_tool"
            }
        else:
            state["messages"].append(AIMessage(content=decision["final_answer"]))
            return {**state, "next_step": "end"}
    except:
        state["messages"].append(AIMessage(content=response.content))
        return {**state, "next_step": "end"}

def tool_execution(state: AgentState) -> AgentState:
    result = tools[state["current_tool"]](state["tool_input"])
    state["messages"].append(FunctionMessage(name=state["current_tool"], content=result))
    state["tool_output"] = result
    state["next_step"] = "continue"
    return state

workflow = StateGraph(AgentState)
workflow.add_node("agent_decision", agent_decision)
workflow.add_node("tool_execution", tool_execution)
workflow.add_conditional_edges("agent_decision", {
    "tool_execution": lambda s: s["next_step"] == "use_tool",
    END: lambda s: s["next_step"] == "end"
})
workflow.add_edge("tool_execution", "agent_decision")
workflow.set_entry_point("agent_decision")
agent = workflow.compile()

result = agent.invoke({
    "messages": [HumanMessage(content="法国的人口是多少？另外，23 * 45是多少？")],
    "next_step": ""
})
for message in result["messages"]:
    print(f"{message.type}: {message.content}")
