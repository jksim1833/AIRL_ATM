from pathlib import Path
import os
import json
import re
import argparse
from openai import OpenAI
import time

class SeatControlChatbot:
    def __init__(self):
        self.client = None
        self.chat_history = []
        self.code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
        self.colors = self.Colors()
        
    class Colors:
        RED = "\033[31m"
        ENDC = "\033[m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"

    def load_openai_api_key(self):
        """tetris_secrets.json 파일에서 OpenAI API 키 로드"""
        possible_paths = [
            Path.home() / "Desktop" / "AIRL_ATM" / "SW" / "tetris_secrets.json",
            Path.home() / "Desktop" / "AIRL_ATM" / "tetris_secrets.json",
            Path("C:/Users/User/Desktop/AIRL_ATM/SW/tetris_secrets.json"),
            Path("C:/Users/User/Desktop/AIRL_ATM/tetris_secrets.json"),
            Path("C:/Users/AIRL/Desktop/Test/AIRL_ATM/SW/tetris_secrets.json")
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
            
            # OpenAI 클라이언트 설정
            if "openai" in secrets and "OPENAI_API_KEY" in secrets["openai"]:
                api_key = secrets["openai"]["OPENAI_API_KEY"]
            elif "OPENAI_API_KEY" in secrets:
                api_key = secrets["OPENAI_API_KEY"]
            else:
                raise KeyError("OpenAI API 키를 찾을 수 없습니다")
            
            self.client = OpenAI(api_key=api_key)
            os.environ["OPENAI_API_KEY"] = api_key
            
            print("🔑 OpenAI API 키 로드 완료")
            
        except KeyError as e:
            raise KeyError(f"🚨 OpenAI API 키를 찾을 수 없습니다: {e}")
        except json.JSONDecodeError:
            raise ValueError("🚨 tetris_secrets.json 파일 형식이 올바르지 않습니다")

    def load_prompts(self, prompt_path, sysprompt_path):
        """프롬프트 파일들 로드"""
        try:
            with open(sysprompt_path, "r", encoding='utf-8') as f:
                sysprompt = f.read()
            
            with open(prompt_path, "r", encoding='utf-8') as f:
                prompt = f.read()
                
            return sysprompt, prompt
        except FileNotFoundError as e:
            print(f"🚨 프롬프트 파일을 찾을 수 없습니다: {e}")
            # 기본 시스템 프롬프트 사용
            sysprompt = "You are a helpful seat control assistant. Provide seat control commands in Python code format."
            prompt = "Initialize seat control system"
            return sysprompt, prompt

    def initialize_chat_history(self, sysprompt):
        """채팅 기록 초기화"""
        self.chat_history = [
            {
                "role": "system",
                "content": sysprompt
            },
            {
                "role": "user",
                "content": "all seat face forward"
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

    def ask(self, prompt):
        """ChatGPT에게 질문하고 응답 받기"""
        self.chat_history.append({
            "role": "user",
            "content": prompt,
        })
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.chat_history,
                temperature=0
            )
            
            content = response.choices[0].message.content
            self.chat_history.append({
                "role": "assistant",
                "content": content,
            })
            
            return content
        except Exception as e:
            print(f"🚨 OpenAI API 호출 중 오류 발생: {e}")
            return "죄송합니다. 응답을 생성할 수 없습니다."

    def extract_python_code(self, content):
        """응답에서 Python 코드 추출"""
        code_blocks = self.code_block_regex.findall(content)
        if code_blocks:
            full_code = "\n".join(code_blocks)
            if full_code.startswith("python"):
                full_code = full_code[7:]
            return full_code
        return None

    def run_chatbot(self, prompt_path="SW/Chain2/main/source/chain2_basic.txt", 
                   sysprompt_path="SW/Chain2/main/source/chain2_system.txt"):
        """챗봇 메인 실행 함수"""
        try:
            # API 키 로드
            print("Initializing ChatGPT...")
            self.load_openai_api_key()
            
            # 프롬프트 로드
            sysprompt, initial_prompt = self.load_prompts(prompt_path, sysprompt_path)
            
            # 채팅 기록 초기화
            self.initialize_chat_history(sysprompt)
            
            print("Initializing Seat Control System...")
            
            # 초기 프롬프트 실행
            self.ask(initial_prompt)
            
            print("Done.")
            print("Welcome to the Seat Control chatbot! I am ready to help you with your seat control questions and commands.")
            print("Commands: !quit or !exit to exit, !clear to clear screen")
            
            # 메인 챗봇 루프
            while True:
                question = input(self.colors.YELLOW + "SeatControl> " + self.colors.ENDC)
                
                if question in ["!quit", "!exit"]:
                    print("챗봇을 종료합니다.")
                    break
                
                if question == "!clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue
                
                if not question.strip():
                    continue
                
                # ChatGPT에게 질문
                response = self.ask(question)
                print(f"\n{response}\n")
                
                # Python 코드 추출 및 실행
                code = self.extract_python_code(response)
                if code is not None:
                    print("Please wait while I execute the seat control commands...")
                    try:
                        exec(code)
                        print("Seat control executed successfully!\n")
                    except Exception as e:
                        print(f"Error executing seat control: {e}\n")
                else:
                    print("No executable code found in the response.\n")
                    
        except Exception as e:
            print(f"🚨 챗봇 실행 중 오류 발생: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Seat Control Chatbot")
    parser.add_argument("--prompt", type=str, 
                       default="SW/Chain2/main/source/chain2_basic.txt",
                       help="Basic prompt file path")
    parser.add_argument("--sysprompt", type=str, 
                       default="SW/Chain2/main/source/chain2_system.txt",
                       help="System prompt file path")
    args = parser.parse_args()
    
    # 챗봇 인스턴스 생성 및 실행
    chatbot = SeatControlChatbot()
    chatbot.run_chatbot(args.prompt, args.sysprompt)


if __name__ == "__main__":
    main()