import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

# 1. 配置DeepSeek模型信息
model_info = {
    "name": "deepseek-chat",
    "parameters": {
        "max_tokens": 2048,
        "temperature": 0.4,
        "top_p": 0.9
    },
    "family": "gpt-4o",  # 必填，保持与原模型一致
    "functions": [],
    "vision": False,
    "json_output": True,
    "function_calling": True  # 若需调用工具，设为True
}
# 2. 创建DeepSeek模型客户端
# DeepSeek API密钥（从环境变量或直接设置）
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "your_key_api")

model_client = OpenAIChatCompletionClient(
    model="deepseek-chat",  # 与官方模型名称一致
    base_url="https://api.deepseek.com",  # DeepSeek API地址
    api_key=deepseek_api_key,
    model_info=model_info
)
# 3. 定义工具函数（可选，如需调用外部工具）
async def get_weather(city: str) -> str:
    """获取城市天气信息（示例函数）"""
    return f"{city}的天气为晴天，温度23°C。"

# 4. 定义专业化智能体（集成DeepSeek模型）
planner_agent = AssistantAgent(
    "planner_agent",
    model_client=model_client,
    description="A helpful assistant that can plan trips.",
    system_message="You are a helpful assistant that can suggest a travel plan for a user based on their request.",
    tools=[get_weather]  # 可选：添加工具函数
)

local_agent = AssistantAgent(
    "local_agent", 
    model_client=model_client,
    description="A local assistant that can suggest local activities or places to visit.",
    system_message="You are a helpful assistant that can suggest authentic and interesting local activities or places to visit for a user and can utilize any context information provided.",
    tools=[get_weather]  # 可选：添加工具函数
)

language_agent = AssistantAgent(
    "language_agent",
    model_client=model_client,
    description="Assists with language translation and cultural nuances.",
    system_message="You are an expert in local languages and cultural norms. Help translate phrases and explain cultural nuances to travelers.",
    tools=[get_weather]  # 可选：添加工具函数
)

travel_summary_agent = AssistantAgent(
    "travel_summary_agent",
    model_client=model_client,
    description="Summarizes travel plans and key information.",
    system_message="You are a concise assistant who can summarize travel plans into key points and provide useful tips for the trip.",
    tools=[get_weather]  # 可选：添加工具函数
)
# 5. 创建协作团队（保持原有逻辑）
termination = TextMentionTermination("TERMINATE")
group_chat = RoundRobinGroupChat(
    [planner_agent, local_agent, language_agent, travel_summary_agent],
    termination_condition=termination
)
# 6. 执行协作任务（保持原有逻辑）

async def main():
    await Console(group_chat.run_stream(task="Plan a 3 day trip to Nepal."))

if __name__ == "__main__":
    asyncio.run(main())
