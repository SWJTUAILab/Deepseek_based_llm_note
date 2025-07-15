from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = llm = ChatOpenAI(model="gpt-4o-mini")
template = """
你是一位{role}。
任务：{task}
请用{style}风格回答以下问题：
{question}
"""

prompt = PromptTemplate(
    input_variables=["role", "task", "style", "question"],
    template=template
)
formatted_prompt = prompt.format(
    role="经验丰富的数学教师",
    task="解释复杂的数学概念",
    style="简洁易懂的",
    question="什么是微积分？"
)
response = model.invoke(formatted_prompt)
print(response)
