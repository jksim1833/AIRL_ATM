from pathlib import Path
import os
import json
import re
import argparse
import google.generativeai as genai
import time

class SeatControlChatbot:
    def __init__(self):
        self.client = None
        self.model = None
        self.chat_session = None
        self.chat_history = []
        self.code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
        self.colors = self.Colors()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    class Colors:
        RED = "\033[31m"
        ENDC = "\033[m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"

    def load_gemini_api_key(self):
        """tetris_secrets.json 파일에서 Google API 키 로드"""
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
            
            # Google API 키 설정
            if "google" in secrets and "GOOGLE_API_KEY" in secrets["google"]:
                api_key = secrets["google"]["GOOGLE_API_KEY"]
            elif "GOOGLE_API_KEY" in secrets:
                api_key = secrets["GOOGLE_API_KEY"]
            else:
                raise KeyError("Google API 키를 찾을 수 없습니다")
            
            # Gemini API 설정
            genai.configure(api_key=api_key)
            os.environ["GOOGLE_API_KEY"] = api_key
            
            # 모델 초기화
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            print("🔑 Google Gemini API 키 로드 및 모델 초기화 완료")
            
        except KeyError as e:
            raise KeyError(f"🚨 Google API 키를 찾을 수 없습니다: {e}")
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
        # Gemini는 시스템 프롬프트를 첫 번째 메시지에 포함
        initial_message = f"{sysprompt}\n\nUser: all seat face forward"
        
        # 초기 응답 예시
        initial_response = """```python
{
    "seat1": ("W", "B", "F"),
    "seat2": ("H", "A", "F"),
    "seat3": ("H", "C", "F"),
    "seat4": ("W", "B", "F")
}
```

This output seat settings to face the seat forward a new direction that is seat_direction("F")."""
        
        # 채팅 세션 시작
        self.chat_session = self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [initial_message]
                },
                {
                    "role": "model",
                    "parts": [initial_response]
                }
            ]
        )

    def extract_python_code(self, content):
        """응답에서 Python 코드 추출"""
        code_blocks = self.code_block_regex.findall(content)
        if code_blocks:
            full_code = "\n".join(code_blocks)
            if full_code.startswith("python"):
                full_code = full_code[7:]
            return full_code
        return None

    def count_tokens(self, text):
        """텍스트의 토큰 수 추정 (Gemini는 정확한 토큰 카운팅 API가 없으므로 추정)"""
        if isinstance(text, str):
            # 대략적인 토큰 수 추정 (영어 기준 4글자당 1토큰, 한글 기준 2글자당 1토큰)
            return len(text.split())
        return 0

    def ask(self, prompt):
        """Gemini에게 질문하고 응답 받기"""
        start_time = time.time()
        
        # 입력 토큰 추정
        input_tokens = self.count_tokens(prompt)
        
        try:
            response = self.chat_session.send_message(prompt)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            content = response.text
            
            # 출력 토큰 추정
            output_tokens = self.count_tokens(content)
            
            # 누적 토큰 업데이트
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            # 메타데이터 출력
            metadata = self.format_metadata(input_tokens, output_tokens, response_time)
            print(metadata)
            
            return content
            
        except Exception as e:
            print(f"🚨 Gemini API 호출 중 오류 발생: {e}")
            return "죄송합니다. 응답을 생성할 수 없습니다."

    def format_metadata(self, input_tokens, output_tokens, response_time):
        """메타데이터를 포맷팅하여 출력"""
        total_tokens = input_tokens + output_tokens
        # Gemini 2.0 Flash 가격 (추정) - 실제 가격은 Google 공식 문서 확인 필요
        cost_input = input_tokens * 0.000075 / 1000  # 추정 비용
        cost_output = output_tokens * 0.0003 / 1000   # 추정 비용
        total_cost = cost_input + cost_output
        
        metadata = f"""
{self.colors.BLUE}📊 토큰 사용량 메타데이터 📊{self.colors.ENDC}
┌─────────────────────────────────────┐
│ 입력 토큰:     {input_tokens:>6} tokens    │
│ 출력 토큰:     {output_tokens:>6} tokens    │
│ 총 토큰:       {total_tokens:>6} tokens    │
│ 응답 시간:     {response_time:>6.2f}s       │
│ 추정 비용:     ${total_cost:>8.6f}      │
└─────────────────────────────────────┘
{self.colors.BLUE}누적 토큰:     {self.total_input_tokens + self.total_output_tokens:>6} tokens{self.colors.ENDC}
"""
        return metadata

    def run_chatbot(self, prompt_path="SW/Chain2/main/source/chain2_basic.txt", 
                   sysprompt_path="SW/Chain2/main/source/chain2_system.txt"):
        """챗봇 메인 실행 함수"""
        try:
            # API 키 로드
            print("Initializing Gemini...")
            self.load_gemini_api_key()
            
            # 프롬프트 로드
            sysprompt, initial_prompt = self.load_prompts(prompt_path, sysprompt_path)
            
            # 채팅 기록 초기화
            self.initialize_chat_history(sysprompt)
            
            print("Initializing Seat Control System...")
            
            # 초기 프롬프트 실행
            self.ask(initial_prompt)
            
            print("Done.")
            print("Welcome to the Seat Control chatbot! I am ready to help you with your seat control questions and commands.")
            print(f"{self.colors.YELLOW}Commands: !quit or !exit to exit, !clear to clear screen, !tokens to show total usage{self.colors.ENDC}")
            
            # 메인 챗봇 루프
            while True:
                question = input(self.colors.YELLOW + "SeatControl> " + self.colors.ENDC)
                
                if question in ["!quit", "!exit"]:
                    # 최종 토큰 사용량 출력
                    total_tokens = self.total_input_tokens + self.total_output_tokens
                    total_cost = (self.total_input_tokens * 0.000075 + self.total_output_tokens * 0.0003) / 1000
                    print(f"\n{self.colors.BLUE}🔚 세션 종료 - 총 사용 토큰: {total_tokens:,} tokens, 총 비용: ${total_cost:.6f}{self.colors.ENDC}")
                    print("챗봇을 종료합니다.")
                    break
                
                if question == "!clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue
                
                if question == "!tokens":
                    total_tokens = self.total_input_tokens + self.total_output_tokens
                    total_cost = (self.total_input_tokens * 0.000075 + self.total_output_tokens * 0.0003) / 1000
                    print(f"\n{self.colors.BLUE}📊 현재까지 총 사용량{self.colors.ENDC}")
                    print(f"입력 토큰: {self.total_input_tokens:,}")
                    print(f"출력 토큰: {self.total_output_tokens:,}")
                    print(f"총 토큰: {total_tokens:,}")
                    print(f"추정 총 비용: ${total_cost:.6f}\n")
                    continue
                
                if not question.strip():
                    continue
                
                # Gemini에게 질문
                response = self.ask(question)
                print(f"\n{self.colors.GREEN}Assistant:{self.colors.ENDC} {response}\n")
                
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
    parser = argparse.ArgumentParser(description="Seat Control Chatbot with Google Gemini")
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