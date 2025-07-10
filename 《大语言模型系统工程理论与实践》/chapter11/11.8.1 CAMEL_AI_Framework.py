from camel.societies import RolePlaying
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dotenv import load_dotenv
import os

# 加载环境变量（API密钥等）
load_dotenv(dotenv_path='.env')

# 初始化DeepSeek模型（保持与原代码一致）
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="deepseek-chat",
    url='https://api.deepseek.com',
    api_key='your-deepseek-api-key'  # 替换为实际API密钥
)

def main(model=model, chat_turn_limit=50) -> None:
    task_prompt = "为股票市场开发一个交易机器人"  # 任务目标
    
    # 创建角色扮演会话（配置角色、模型、任务）
    role_play_session = RolePlaying(
        assistant_role_name="Python 程序员",
        assistant_agent_kwargs=dict(model=model),
        user_role_name="股票交易员",
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
        output_language='中文'
    )

    # 打印系统消息及任务提示（移除颜色标记）
    print(f"AI 助手系统消息:\n{role_play_session.assistant_sys_msg}\n")
    print(f"AI 用户系统消息:\n{role_play_session.user_sys_msg}\n")
    print(f"原始任务提示:\n{task_prompt}\n")
    print(f"指定的任务提示:\n{role_play_session.specified_task_prompt}\n")
    print(f"最终任务提示:\n{role_play_session.task_prompt}\n")

    n = 0
    input_msg = role_play_session.init_chat()
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)

        # 终止条件检查（保持原逻辑）
        if assistant_response.terminated:
            print(f"AI 助手已终止。原因: {assistant_response.info['termination_reasons']}.")
            break
        if user_response.terminated:
            print(f"AI 用户已终止。原因: {user_response.info['termination_reasons']}.")
            break

        # 基础打印交互内容（移除动画和颜色）
        print(f"AI 用户:\n\n{user_response.msg.content}\n")
        print(f"AI 助手:\n\n{assistant_response.msg.content}\n")

        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break
        input_msg = assistant_response.msg

if __name__ == "__main__":
    main()
