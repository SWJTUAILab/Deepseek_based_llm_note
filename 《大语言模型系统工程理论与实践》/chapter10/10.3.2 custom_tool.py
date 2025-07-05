from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式，如 '23 * 45'")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "用于执行数学计算的工具"
    args_schema = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            return str(eval(expression))
        except Exception as e:
            return f"计算错误: {str(e)}"
