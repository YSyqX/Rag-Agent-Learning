import os
from typing import List, Literal
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


load_dotenv()

model = init_chat_model( 
    model = "openai:qwen3.8-flash",
    api_key = os.getenv("DASHSCOPE_API_KEY"), # type: ignore
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄",le=120,ge=0)
    gender: Literal["男", "女"] = Field(description="性别")
    city: str = Field(description="城市")
    character: List[str] = Field(description="性格特点")


model_with_structured = model.with_structured_output(person)
res = model_with_structured.invoke("请给我一份个人信息")
print(res)
print(type(res))