from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI()
prompt = [HumanMessage('프랑스의 수도는 어디인가요?')]

result = model.invoke(prompt)

print(result)          # AIMessage 객체
print("-"*30)
print(result.content)  # 실제 텍스트만