from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from rich import print as rprint
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

load_dotenv()



model1 = ChatOpenAI(
    model = "qwen3.8-flash",
    api_key = os.getenv("DASHSCOPE_API_KEY"), # type: ignore
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

model2 = ChatOpenAI(
    model = "deepseek-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"), # type: ignore
    base_url="https://api.deepseek.com",
)

model_fall_back = ModelFallbackMiddleware(model2)

agent = create_agent(
    model = model1,
    tools = [],
    system_prompt = "你是问题回答助手Ysyq,请合理使用工具来回答用户的问题.",
    middleware = [model_fall_back]
)
