from dotenv import load_dotenv
load_dotenv()

from langchain_openai.llms import OpenAI

model = OpenAI(model='gpt-3.5-turbo-instruct')

resp = model.invoke('하늘이')

print(resp)