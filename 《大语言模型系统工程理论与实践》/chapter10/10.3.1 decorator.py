from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """搜索网络获取信息。"""
    return f"搜索结果：关于 {query} 的相关信息..."
print(search_web.invoke("langchain框架"))
