import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

print("脚本启动，API Key:", bool(os.getenv("DASHSCOPE_API_KEY")))

model = ChatTongyi(model="qwen-max")  # type: ignore # 修正模型名

messages = [
    SystemMessage(content="你是一个边塞诗人。"),
    HumanMessage(content="写一首唐诗"),
    AIMessage(content="大漠孤烟直，长河落日圆。"),
    HumanMessage(content="按照你上一个回复的格式，再写一首唐诗。")
]

print("正在调用 invoke...")
try:
    response = model.invoke(messages)
    print("完整响应:", response.content)
except Exception as e:
    print("错误:", e)
    import traceback
    traceback.print_exc()