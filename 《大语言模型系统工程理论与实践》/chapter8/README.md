# 第八章代码说明

### 1.  系统操作环境

- 操作系统: Windows 11
- CPU: AMD Ryzen 9 7945HX
- GPU: NVDIA GeForce RTX 4060 8G

### 2.  需要安装的依赖

langchain==0.3.25

langhcain_openai==0.3.16

promptflow==1.18.0

promptflow-tools==1.6.0

openai==1.78.1

### 3.  运行结果及解释（屏幕截图）

### 3.1 提示词模板

**8.3.1 template_test.py**

a.  根据业务用途，构造特定的提示词模板

b.  依赖：langchain，langhcain_openai

c.   运行：python template.py

![image-20250705104311887](C:\Users\32601\AppData\Roaming\Typora\typora-user-images\image-20250705104311887.png)

### 3.2 提示词版本控制与协作工具

**8.3.2 analyze_query.py**

a.  使用PromptFlow的@tool装饰器，注册一个可以集成在流程图中的工具节点。

b.  依赖：promptflow，promptflow-tools，openai

c.   注意：需要提前将OPENAI_API_KEY和OPENAI_API_BASE配置在环境变量中。



**8.3.3 generate_response.py**

a.  使用PromptFlow的@tool装饰器，注册一个可以集成在流程图中的工具节点。

b.  依赖：promptflow，promptflow-tools，openai

c.   注意：需要提前将OPENAI_API_KEY和OPENAI_API_BASE配置在环境变量中。



**8.3.4 flow.yaml**

a.  定义数据流在多个工具节点之间传递。

b.  依赖：无

c.   注意：该文件为配置文件



**8.3.5 promptflow_test.py**

a.  通过输入问题，调用flow并执行flow，并根据定义好的提示词模板给出回答

b.  依赖：promptflow

c.   运行：python promptflow.py

![image-20250705104408266](C:\Users\32601\AppData\Roaming\Typora\typora-user-images\image-20250705104408266.png)

### 3.3 思维链增强工具

**8.3.6 tracing_test.py**

a.   通过langchain框架构建一个链，用于追踪token的使用情况。

b.   依赖：langchain

c.   运行：python tracing.py

![image-20250705104454960](C:\Users\32601\Desktop\工位备份\组内文档\教材\第八章代码验证过程技术报告.assets\image-20250705104454960.png)

![image-20250705104505953](C:\Users\32601\Desktop\工位备份\组内文档\教材\第八章代码验证过程技术报告.assets\image-20250705104505953.png)

### 4.  问题记录

所有代码都已成功验证运行，但存在以下注意事项:

所有代码的运行均需要提前在环境变量中配置OPENAI_API_KEY和OPENAI_API_BASE，否则无法使用模型。