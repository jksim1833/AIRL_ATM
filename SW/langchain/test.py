# c:\Users\User\Desktop\AIRL_ATM\SW\langchain\test.py
from dotenv import load_dotenv; load_dotenv()

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


from dotenv import load_dotenv; load_dotenv()

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

async def main():
    print("RUNNING:", __file__)

    chat = ChatOpenAI(model="gpt-4o", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
    ("system", "Act as a senior software engineer at a startup company."),
    ("user", "{q}")
    ])
    chain = prompt | chat | StrOutputParser()
    texts = await chain.abatch([
        {"q": "Explain what LangChain is in 2 concise sentences."}
    ] * 2)
    print("\n\n".join(texts))

if __name__ == "__main__":
    asyncio.run(main())