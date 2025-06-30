import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_community.chat_models import ChatOpenAI
from IPython.display import Image, display

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def route(state: AgentState) -> Literal["search", "calculate", END]:
    last_msg = state["messages"][-1]["content"]
    if "搜索" in last_msg:
        return "search"
    elif "计算" in last_msg:
        return "calculate"
    else:
        return END

def search(state: AgentState) -> dict:
    query = state["messages"][-1]["content"]
    return {"messages": [{"role": "function", "name": "search", "content": f"搜索结果：关于 {query} 的信息..."}]}

def calculate(state: AgentState) -> dict:
    return {"messages": [{"role": "function", "name": "calculate", "content": "计算结果：42"}]}

def generate_response(state: AgentState) -> dict:
    llm = ChatOpenAI()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("route", route)
workflow.add_node("search", search)
workflow.add_node("calculate", calculate)
workflow.add_node("generate_response", generate_response)
workflow.add_edge(START, "route")
workflow.add_conditional_edges("route", route, {
    "search": "search",
    "calculate": "calculate",
    END: END,
})
workflow.add_edge("search", "generate_response")
workflow.add_edge("calculate", "generate_response")
workflow.add_edge("generate_response", "route")
app = workflow.compile()
with open("graph.png", "wb") as f:
    f.write(app.get_graph().draw_png())
display(Image(filename="graph.png"))
