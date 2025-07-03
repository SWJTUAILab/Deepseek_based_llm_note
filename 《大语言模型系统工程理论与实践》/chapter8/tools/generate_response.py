# tools/generate_response.py
from promptflow.core import tool
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

@tool
def generate_response(analysis: str) -> str:
    """根据分析生成回答"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"基于以下分析生成回答：{analysis},回答的结果中，在末尾加上回答完毕"}]
    )
    return response.choices[0].message.content