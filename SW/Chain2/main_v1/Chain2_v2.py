from pathlib import Path
import os, json, re, argparse
import openai
import math
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="SW/Chain2/main/source/chain2_basic.txt")
parser.add_argument("--sysprompt", type=str, default="SW/Chain2/main/source/chain2_system.txt")
args = parser.parse_args()

def _load_openai_api_key(self):
    """tetris_secrets.json 파일에서 OpenAI API 키 로드"""
    # 여러 위치에서 tetris_secrets.json 파일 탐색
    possible_paths = [
        Path.home() / "Desktop" / "AIRL_ATM" / "SW" / "tetris_secrets.json",
        Path.home() / "Desktop" / "AIRL_ATM" / "tetris_secrets.json",
        Path("C:/Users/User/Desktop/AIRL_ATM/SW/tetris_secrets.json"),
        Path("C:/Users/User/Desktop/AIRL_ATM/tetris_secrets.json")
    ]
    
    secrets_path = None
    for path in possible_paths:
        print(f"🔍 탐색 중: {path}")
        if path.exists():
            secrets_path = path
            print(f"✅ 파일 발견: {secrets_path}")
            break
    
    if not secrets_path:
        print("❌ 다음 위치에서 tetris_secrets.json 파일을 찾을 수 없습니다:")
        for path in possible_paths:
            print(f"  - {path}")
        raise FileNotFoundError("🚨 tetris_secrets.json 파일을 찾을 수 없습니다.")
    
    try:
        with open(secrets_path, 'r', encoding='utf-8') as f:
            secrets = json.load(f)
        
        # OpenAI API 키만 환경 변수로 설정
        os.environ["OPENAI_API_KEY"] = secrets["openai"]["OPENAI_API_KEY"]
        
        print("🔑 OpenAI API 키 로드 완료")
        
    except KeyError as e:
        raise KeyError(f"🚨 OpenAI API 키를 찾을 수 없습니다: {e}")
    except json.JSONDecodeError:
        raise ValueError("🚨 tetris_secrets.json 파일 형식이 올바르지 않습니다")




with open('C:/Users/AIRL/Desktop/Test/AIRL_ATM/SW/tetris_secrets.json') as f:
    credentials = json.load(f)

print("Initializing ChatGPT...")
openai.api_key = credentials["OPENAI_API_KEY"]

with open(args.sysprompt, "r") as f:
    sysprompt = f.read()

chat_history = [
    {
        "role": "system",
        "content": sysprompt
    },
    {
        "role": "user",
        "content": "all seat face foward"
    },
    {
        "role": "assistant",
        "content": """```python
{
    "seat1": ("W", "B", "F"),
    "seat2": ("H", "A", "F"),
    "seat3": ("H", "C", "F"),
    "seat4": ("W", "B", "F")
}
```

This output seat settings to face the seat forward a new direction that is seat_direction("F")."""
    }
]


def ask(prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return chat_history[-1]["content"]


print(f"Done.")

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)


def extract_python_code(content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None


class colors:  # You may need to change color settings
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"


print(f"Initializing Seat Control System...")
print(f"Done.")

with open(args.prompt, "r") as f:
    prompt = f.read()

ask(prompt)
print("Welcome to the Seat Control chatbot! I am ready to help you with your seat control questions and commands.")

while True:
    question = input(colors.YELLOW + "SeatControl> " + colors.ENDC)

    if question == "!quit" or question == "!exit":
        break

    if question == "!clear":
        os.system("cls")
        continue

    response = ask(question)

    print(f"\n{response}\n")

    code = extract_python_code(response)
    if code is not None:
        print("Please wait while I execute the seat control commands...")
        try:
            # 좌석 제어 명령 실행
            exec(code)
            print("Seat control executed successfully!\n")
        except Exception as e:
            print(f"Error executing seat control: {e}\n")
    else:
        print("No executable code found in the response.\n")