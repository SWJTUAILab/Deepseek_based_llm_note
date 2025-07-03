from langchain.tools import StructuredTool
from pydantic import BaseModel

class WeatherInput(BaseModel):
    location: str
    date: str = None

def get_weather(location: str, date: str = None) -> str:
    return f"{location}的天气预报：晴，气温25°C"

weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="weather",
    description="获取指定地点和日期的天气预报",
    args_schema=WeatherInput
)
