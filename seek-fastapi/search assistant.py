from contextlib import asynccontextmanager
import logging
from pathlib import Path
from tabnanny import check
from typing import Any
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from rich import print as rprint
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from fastapi import FastAPI
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# 创建日志记录器
# %(asctime)s: 日志记录时间 %(levelname)s: 日志级别 %(filename)s: 文件名 %(lineno)d: 行号 %(message)s: 日志信息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s"  # 日志格式
)

# 定义模型
model_qwen = init_chat_model( 
    model = "openai:qwen3.8-flash",
    api_key = os.getenv("DASHSCOPE_API_KEY"), # type: ignore
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

model_deepseek = init_chat_model(
    model = "openai:deepseek-v4-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"), # type: ignore
    base_url="https://api.deepseek.com",
)

# 定义中间件
# 1.备用模型
model_fall_back = ModelFallbackMiddleware(model_deepseek)

# 创建检查点
checkpoint = InMemorySaver()

async def limit_tavily_results(request, handler):
    if request.server_name == "tavily" and "search" in request.name.lower():
        request = request.override(
            args={**request.args, "max_results": 3}
        )
    return await handler(request)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ===== 启动时执行 =====
    logging.info("创建agent")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    client = MultiServerMCPClient({
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}",
        }}, 
        tool_interceptors=[limit_tavily_results]
        )

    all_tools = await client.get_tools()

    agent = create_agent(
        model=model_qwen,
        tools=all_tools,
        middleware=[model_fall_back],
        checkpointer=checkpoint,
        system_prompt="你是问题回答助手Ysyq,请合理使用工具来回答用户的问题."
    )  # type: ignore

    # 存到 app.state，全局可访问
    app.state.agent = agent
    app.state.mcp_client = client

    yield  # 应用运行期间


app = FastAPI(lifespan=lifespan)

# 创建静态文件目录
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 数据模型
class ApiResponse(BaseModel):
    code: int
    message: str
    data: Any

class ChatRequest(BaseModel):
    session_id: str
    message: str

# 定义根路径的处理函数
@app.get("/")
def root():
    logging.info("访问根路径")
    return FileResponse(STATIC_DIR / "index.html")

# 新建会话
@app.post("/api/new_session")
def new_session() -> ApiResponse:
    logging.info("新建会话")
    session_id = os.urandom(16).hex()  # 生成随机会话ID
    return ApiResponse(code=200, message="新建会话成功", data={"session_id": session_id})

# 与agent对话
@app.post("/api/chat")
async def chat(request: ChatRequest) -> ApiResponse:
    logging.info(f"与AI交互:{request.session_id}-{request.message}")

    # 获取agent
    agent = app.state.agent

    # 通过thread_id区分对话
    thread_config = {
        "configurable": {
        "thread_id": request.session_id
    }}

    res =await agent.ainvoke(
        {"messages":[HumanMessage(request.message)]},
        config = thread_config
        )

    return ApiResponse(code=200,message="交互成功",data=res["messages"][-1].content)

# 加载会话
@app.get("/api/sessions/{session_id}")
def load_session(session_id:str) -> ApiResponse:
    logging.info(f"加载会话:{session_id}")

    # 获取agent
    agent = app.state.agent

    # 通过thread_id区分对话
    thread_config = {
         "configurable": {
        "thread_id": session_id
    }}
    
    state = agent.get_state(thread_config)
    return ApiResponse(code=200,message="加载会话成功",data=state.values.get("messages",[]))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)