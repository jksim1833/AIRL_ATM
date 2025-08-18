from google import genai

# 🔑 API 키 환경변수에 설정해 두었으면 생략 가능
client = genai.Client(api_key="")

# 지원되는 모델 리스트 확인
models = client.models.list()

for m in models:
    print(m.name)
