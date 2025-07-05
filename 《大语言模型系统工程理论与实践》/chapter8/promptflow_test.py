from promptflow.core import Flow

# 加载 flow
flow = Flow.load("flow.yaml")

# 调用 flow
result = flow(query="如何提高编程效率？")
print(result["answer"])