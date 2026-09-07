import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from rich import print as rprint


load_dotenv()

# print("Initializing chat models...")

# model = init_chat_model( 
#     model = "openai:qwen3.8-flash",
#     api_key = os.getenv("DASHSCOPE_API_KEY"), # type: ignore
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# print("Chat model initialized successfully.")
# print(model.invoke("请写一首唐诗").content)

model2 = init_chat_model(
    model = "openai:deepseek-v4-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"), # type: ignore
    base_url="https://api.deepseek.com",
)
res = model2.invoke("请写一首唐诗")
print(res.content)
rprint(res)