# 第十一章代码验证

## 系统操作环境

  * **操作系统** ：Windows 11
  * **CPU** ：Intel(R)Core(TM) i7-14650HX
  * **GPU** ：NVDIA GeForce RTX 4060 8G
  * **Python** ：3.11.6

## 依赖安装

  1. 安装 autogenstudio：`pip install -U "autogenstudio"`
  2. 安装 camel-ai：`pip install camel-ai`

## 代码及运行结果解释

### 11.5.1 Autogen 框架示例

  * **代码** ：
```python
import asyncio
from autogen import Console, group_chat

async def main():
    await Console(group_chat.run_stream(task="Plan a 3 day trip to Nepal."))

if __name__ == "__main__":
    asyncio.run(main())
```

  * **解释** ：该代码通过 Autogen 框架创建了一个控制台对象，并使用 group_chat.run_stream() 方法来规划一个为期三天的尼泊尔之旅。在 async 和 await 的配合下，实现了异步任务的执行，能够更高效地处理任务。

## 问题记录

  1. **Autogen 安装问题** ：原代码中直接使用 `pip install ag2` 可能无法满足官方推荐的库版本，需采用 `pip install -U "autogenstudio"` 进行安装。
  2. **代码兼容性问题** ：原代码中的 `await Console(group_chat.run_stream(task="Plan a 3 day trip to Nepal."))` 在 Python 下运行不通，需修改为 async 函数封装的形式。
