from dotenv import load_dotenv
import os

load_dotenv()  # 또는 load_dotenv(dotenv_path="C:/경로/.env")

print("API KEY:", os.getenv("OPENAI_API_KEY"))  # 확인용

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

print("RUNNING:", __file__)  # 실행한 파일 경로 확인용

chat = ChatOpenAI(model="gpt-4o", temperature=0.3)

messages = [
    SystemMessage(content="Act as a senior software engineer at a startup company."),
    HumanMessage(content="Explain what LangChain is in 2 concise sentences.")
]

synchronous_llm_result = chat.batch([messages] * 2)

for i, resp in enumerate(synchronous_llm_result, start=1):
    print(f"=== 응답 {i} ===")
    print(resp.content)