from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

model1 = ChatOpenAI(
    model = "qwen3.8-flash",
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

model2 = ChatOpenAI(
    model = "deepseek-v4-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# print(model1.invoke("你好").content)
# print(model2.invoke("你好").content)
res = model1.stream("请写一首唐诗")
for chunk in res:
    print(chunk.content, end="")
res = model2.stream("请写一首唐诗")
print("\n")
for chunk in res:
    print(chunk.content, end="")