# start.py
from dotenv import load_dotenv
load_dotenv()  # .env에서 환경변수 읽기

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1) 키 로드 & 정리
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 비어 있습니다. .env 또는 환경변수를 확인하세요.")
api_key = api_key.strip()  # 붙어 들어간 공백/개행 제거
os.environ["OPENAI_API_KEY"] = api_key  # 현재 프로세스에 확정 세팅

# 2) 프로젝트 키 사용 시, ORG/PROJECT 관련 충돌 방지
#    - 예전에 잡아둔 OPENAI_ORG_ID / OPENAI_ORGANIZATION 등이 있으면 제거(이 프로세스 한정)
for k in ("OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT"):
    if os.getenv(k):
        os.environ.pop(k, None)

# 3) LLM 생성 (project 인자 쓰지 않음)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key,  # 명시 전달
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "넌 간결하게 답하는 도우미야."),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    print(chain.invoke({"input": "LangChain에서 인사 한마디 해줘."}))
