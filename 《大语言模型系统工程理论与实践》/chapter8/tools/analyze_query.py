# tools/analyze_query.py
from promptflow.core import tool
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

@tool
def analyze_query(query: str) -> str:
    """分析用户查询意图"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"分析以下查询的意图和关键要素：{query}"}]
    )
    return response.choices[0].message.content
