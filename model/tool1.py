from langchain_community.tools import tool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, ToolMessage,SystemMessage,HumanMessage,AIMessage
load_dotenv()

class WeatherArgs(BaseModel):
    location: str = Field(description="指定位置",default="杭州")


@tool(args_schema=WeatherArgs)
def get_weather1(location: str):
    """
    获取今天指定位置的天气信息
    """
    return f"{location}天气晴朗."

@tool(args_schema=WeatherArgs)
def get_weather2(location: str):
    """
    获取明天指定位置的天气信息
    """
    return f"{location}下雨."

model = ChatOpenAI(
    model = "qwen3.8-flash",
    api_key = os.getenv("DASHSCOPE_API_KEY"), # type: ignore
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 维护历史记录
messages:list[BaseMessage] = [
    SystemMessage(content="你是一个天气助手。"),
    HumanMessage(content="请告诉我今天和明天杭州的天气。")
]

# 绑定工具
model_with_tools = model.bind_tools([get_weather1, get_weather2])

# 调用工具
while True:
    response = model_with_tools.invoke(messages)
    if response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            if tool_name == "get_weather1":
                weather_info = get_weather1.invoke(tool_call)
            elif tool_name == "get_weather2":
                weather_info = get_weather2.invoke(tool_call)
            tool_message = ToolMessage(content=weather_info, tool_call_id=tool_call["id"])
            messages.append(tool_message)
    else:
        # print("最终响应:", response)
        messages.append(response)
        break

# 打印对话历史
for msg in messages:
    msg.pretty_print()
