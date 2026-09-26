from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from rich import print as rprint
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


# 定义模型
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


# 定义工具
# 1.搜索工具tavily
search_tool = TavilySearch(
    max_results=3,
    topic="general",
    search_depth = "advanced"
)


# 定义中间件
# 1.备用模型
model_fall_back = ModelFallbackMiddleware(model2)


# 创建检查点
checkpoint = InMemorySaver()


# 构建agent
agent = create_agent(
    model = model1,
    tools = [search_tool],
    system_prompt = "你是问题回答助手Ysyq,请合理使用工具来回答用户的问题.",
    middleware = [model_fall_back],
    checkpointer = checkpoint
)

# 通过thread_id区分对话
thread_config = {
    "configurable": {
        "thread_id": "user1"
    }
}

# 调用agent进行多轮对话
while True:
    user_input = input("用户输入: ")
    if user_input.lower() == "exit":
        break
    res = agent.invoke(
        {"messages":[HumanMessage(content=user_input)]},
        config=thread_config # type: ignore
    )
    print(res["messages"][-1].content)
    print("-"*50)