from pathlib import Path
import os
import json
import re
import argparse
import google.generativeai as genai
from openai import OpenAI
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
        self.api_type = None  # 'gpt' or 'gemini'
        
    class Colors:
        RED = "\033[31m"
        ENDC = "\033[m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"

    def select_api_and_model(self):
        """API 타입 선택 (Gemini 또는 GPT-4o만)"""
        print(f"{self.colors.CYAN}🤖 AI API 선택{self.colors.ENDC}")
        print("1. Google Gemini 2.0 Flash")
        print("2. OpenAI GPT-4o")
        
        while True:
            try:
                choice = int(input(f"{self.colors.YELLOW}API를 선택하세요 (1 또는 2): {self.colors.ENDC}"))
                if choice == 1:
                    self.api_type = 'gemini'
                    print(f"{self.colors.GREEN}✅ Google Gemini 2.0 Flash 선택됨{self.colors.ENDC}")
                    break
                elif choice == 2:
                    self.api_type = 'gpt'
                    print(f"{self.colors.GREEN}✅ OpenAI GPT-4o 선택됨{self.colors.ENDC}")
                    break
                else:
                    print(f"{self.colors.RED}❌ 1 또는 2를 입력해주세요.{self.colors.ENDC}")
            except ValueError:
                print(f"{self.colors.RED}❌ 숫자를 입력해주세요.{self.colors.ENDC}")

    def load_api_keys(self):
        """API 키 로드 및 클라이언트 초기화"""
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
            
            if self.api_type == 'gpt':
                self._initialize_gpt(secrets)
            elif self.api_type == 'gemini':
                self._initialize_gemini(secrets)
                
        except KeyError as e:
            raise KeyError(f"🚨 API 키를 찾을 수 없습니다: {e}")
        except json.JSONDecodeError:
            raise ValueError("🚨 tetris_secrets.json 파일 형식이 올바르지 않습니다")

    def _initialize_gpt(self, secrets):
        """OpenAI GPT-4o 초기화"""
        # OpenAI API 키 찾기
        if "openai" in secrets and "OPENAI_API_KEY" in secrets["openai"]:
            api_key = secrets["openai"]["OPENAI_API_KEY"]
        elif "OPENAI_API_KEY" in secrets:
            api_key = secrets["OPENAI_API_KEY"]
        else:
            raise KeyError("OpenAI API 키를 찾을 수 없습니다")
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=api_key)
        os.environ["OPENAI_API_KEY"] = api_key
        
        print(f"🔑 OpenAI GPT-4o API 키 로드 및 클라이언트 초기화 완료")

    def _initialize_gemini(self, secrets):
        """Google Gemini 2.0 Flash 초기화"""
        # Google API 키 찾기
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
        
        print("🔑 Google Gemini 2.0 Flash API 키 로드 및 모델 초기화 완료")

    def load_prompts(self, prompt_path, sysprompt_path, userprompt_path):
        """프롬프트 파일들 로드"""
        try:
            with open(sysprompt_path, "r", encoding='utf-8') as f:
                sysprompt = f.read()
            
            with open(prompt_path, "r", encoding='utf-8') as f:
                prompt = f.read()
                
            with open(userprompt_path, "r", encoding='utf-8') as f:
                userprompt = f.read()
                
            return sysprompt, prompt, userprompt
        except FileNotFoundError as e:
            print(f"🚨 프롬프트 파일을 찾을 수 없습니다: {e}")
            # 기본 프롬프트 사용
            sysprompt = "You are a helpful seat control assistant. Provide seat control commands in JSON format."
            prompt = "Initialize seat control system"
            userprompt = "{}"  # 기본 JSON 형태
            return sysprompt, prompt, userprompt

    def initialize_chat_history(self, sysprompt, userprompt):
        """채팅 기록 초기화 - API 타입에 따라 분기"""
        if self.api_type == 'gemini':
            self._initialize_gemini_chat(sysprompt, userprompt)
        elif self.api_type == 'gpt':
            self._initialize_gpt_chat(sysprompt, userprompt)

    def _initialize_gemini_chat(self, sysprompt, userprompt):
        """Gemini 채팅 초기화"""
        # Gemini는 시스템 프롬프트를 첫 번째 메시지에 포함
        initial_message = f"{sysprompt}\n\nUser: all seat face forward"
        
        # 초기 응답 예시 - 순수 JSON 형식
        initial_response = """{
  "cell_layout": {
    "1": { "rail": "X", "pos": "A", "face": "F", "mode": "chair" },
    "2": { "rail": "Y", "pos": "B", "face": "F", "mode": "chair" },
    "3": { "rail": "X", "pos": "C", "face": "F", "mode": "chair" },
    "4": { "rail": "Y", "pos": "A", "face": "F", "mode": "chair" }
  }
}

This output seat settings to face all seats forward with the new cell_layout structure."""
        
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
                },
                {
                    "role": "user",
                    "parts": [userprompt]
                }
            ]
        )

    def _initialize_gpt_chat(self, sysprompt, userprompt):
        """GPT-4o 채팅 초기화"""
        # GPT는 시스템 메시지와 초기 대화를 별도로 구성
        self.chat_history = [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": "all seat face forward"},
            {"role": "assistant", "content": """{
  "cell_layout": {
    "1": { "rail": "X", "pos": "A", "face": "F", "mode": "chair" },
    "2": { "rail": "Y", "pos": "B", "face": "F", "mode": "chair" },
    "3": { "rail": "X", "pos": "C", "face": "F", "mode": "chair" },
    "4": { "rail": "Y", "pos": "A", "face": "F", "mode": "chair" }
  }
}

This output seat settings to face all seats forward with the new cell_layout structure."""},
            {"role": "user", "content": userprompt}
        ]

    def extract_python_code(self, content):
        """응답에서 Python 코드 추출"""
        code_blocks = self.code_block_regex.findall(content)
        if code_blocks:
            full_code = "\n".join(code_blocks)
            if full_code.startswith("python"):
                full_code = full_code[7:]
            return full_code
        return None

    def count_tokens_estimate(self, text):
        """텍스트의 토큰 수 추정 (정확하지 않지만 대략적인 추정)"""
        if isinstance(text, str):
            # 대략적인 토큰 수 추정 (영어 기준 4글자당 1토큰, 한글 기준 2글자당 1토큰)
            return len(text.split())
        return 0

    def ask(self, prompt):
        """API 타입에 따라 분기하여 질문하고 응답 받기"""
        if self.api_type == 'gpt':
            return self._ask_gpt(prompt)
        elif self.api_type == 'gemini':
            return self._ask_gemini(prompt)
        else:
            raise ValueError("API 타입이 설정되지 않았습니다.")

    def _ask_gpt(self, prompt):
        """GPT-4o에게 질문하고 응답 받기"""
        start_time = time.time()
        
        try:
            # 채팅 히스토리에 사용자 메시지 추가
            self.chat_history.append({"role": "user", "content": prompt})
            
            # OpenAI API 호출 (GPT-4o 고정)
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.chat_history,
                temperature=0.7,
                max_tokens=2000
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # 응답 추출
            content = response.choices[0].message.content
            
            # 채팅 히스토리에 응답 추가
            self.chat_history.append({"role": "assistant", "content": content})
            
            # 토큰 사용량 업데이트
            input_tokens = 0
            output_tokens = 0
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
            
            # 메타데이터 출력
            metadata = self.format_gpt_metadata(input_tokens, output_tokens, response_time)
            print(metadata)
            
            return content
            
        except Exception as e:
            print(f"🚨 GPT-4o API 호출 중 오류 발생: {e}")
            return "죄송합니다. 응답을 생성할 수 없습니다."

    def _ask_gemini(self, prompt):
        """Gemini에게 질문하고 응답 받기"""
        start_time = time.time()
        
        # 입력 토큰 추정
        input_tokens = self.count_tokens_estimate(prompt)
        
        try:
            response = self.chat_session.send_message(prompt)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            content = response.text
            
            # 출력 토큰 추정 또는 실제 사용량 사용
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', input_tokens)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            else:
                output_tokens = self.count_tokens_estimate(content)
            
            # 누적 토큰 업데이트
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            # 메타데이터 출력
            metadata = self.format_gemini_metadata(input_tokens, output_tokens, response_time)
            print(metadata)
            
            return content
            
        except Exception as e:
            print(f"🚨 Gemini API 호출 중 오류 발생: {e}")
            return "죄송합니다. 응답을 생성할 수 없습니다."

    def format_gpt_metadata(self, input_tokens, output_tokens, response_time):
        """GPT-4o 메타데이터를 포맷팅하여 출력"""
        total_tokens = input_tokens + output_tokens
        
        # GPT-4o 가격 (2024년 기준)
        cost_input = input_tokens * 0.00250 / 1000
        cost_output = output_tokens * 0.01000 / 1000
        total_cost = cost_input + cost_output
        
        metadata = f"""
{self.colors.BLUE}📊 GPT-4o 토큰 사용량 메타데이터 📊{self.colors.ENDC}
┌─────────────────────────────────────┐
│ 입력 토큰:     {input_tokens:>6} tokens    │
│ 출력 토큰:     {output_tokens:>6} tokens    │
│ 총 토큰:       {total_tokens:>6} tokens    │
│ 응답 시간:     {response_time:>6.2f}s       │
│ 실제 비용:     ${total_cost:>8.6f}      │
└─────────────────────────────────────┘
{self.colors.BLUE}누적 토큰:     {self.total_input_tokens + self.total_output_tokens:>6} tokens{self.colors.ENDC}
"""
        return metadata

    def format_gemini_metadata(self, input_tokens, output_tokens, response_time):
        """Gemini 메타데이터를 포맷팅하여 출력"""
        total_tokens = input_tokens + output_tokens
        # Gemini 2.0 Flash 가격 (추정) - 실제 가격은 Google 공식 문서 확인 필요
        cost_input = input_tokens * 0.000075 / 1000  # 추정 비용
        cost_output = output_tokens * 0.0003 / 1000   # 추정 비용
        total_cost = cost_input + cost_output
        
        metadata = f"""
{self.colors.BLUE}📊 Gemini 2.0 Flash 토큰 사용량 메타데이터 📊{self.colors.ENDC}
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
                   sysprompt_path="SW/Chain2/main/source/chain2_system.txt",
                   userprompt_path="SW/Chain2/main_v2/source/chain2_option_list.json"):
        """챗봇 메인 실행 함수"""
        try:
            # API 및 모델 선택
            print(f"{self.colors.MAGENTA}{'='*50}{self.colors.ENDC}")
            print(f"{self.colors.MAGENTA}🚀 Seat Control Chatbot 초기화{self.colors.ENDC}")
            print(f"{self.colors.MAGENTA}{'='*50}{self.colors.ENDC}")
            
            self.select_api_and_model()
            
            # API 키 로드
            api_name = "GPT-4o" if self.api_type == 'gpt' else "Gemini 2.0 Flash"
            print(f"Initializing {api_name}...")
            self.load_api_keys()
            
            # 프롬프트 로드 (3개 파일)
            sysprompt, initial_prompt, userprompt = self.load_prompts(prompt_path, sysprompt_path, userprompt_path)
            
            # 채팅 기록 초기화
            self.initialize_chat_history(sysprompt, userprompt)
            
            print("Initializing Seat Control System...")
            
            # 초기 프롬프트 실행
            self.ask(initial_prompt)
            
            print("Done.")
            print(f"Welcome to the Seat Control chatbot! I am ready to help you with your seat control questions and commands using {api_name}.")
            print(f"{self.colors.YELLOW}Commands: !quit or !exit to exit, !clear to clear screen, !tokens to show total usage{self.colors.ENDC}")
            
            # 메인 챗봇 루프
            while True:
                question = input(self.colors.YELLOW + "SeatControl> " + self.colors.ENDC)
                
                if question in ["!quit", "!exit"]:
                    # 최종 토큰 사용량 출력
                    total_tokens = self.total_input_tokens + self.total_output_tokens
                    
                    if self.api_type == 'gpt':
                        # GPT-4o 가격
                        total_cost = (self.total_input_tokens * 0.00250 + self.total_output_tokens * 0.01000) / 1000
                    else:  # gemini
                        total_cost = (self.total_input_tokens * 0.000075 + self.total_output_tokens * 0.0003) / 1000
                    
                    print(f"\n{self.colors.BLUE}🔚 세션 종료 - 총 사용 토큰: {total_tokens:,} tokens, 총 비용: ${total_cost:.6f}{self.colors.ENDC}")
                    print("챗봇을 종료합니다.")
                    break
                
                if question == "!clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue
                
                if question == "!tokens":
                    total_tokens = self.total_input_tokens + self.total_output_tokens
                    
                    if self.api_type == 'gpt':
                        # GPT-4o 가격
                        total_cost = (self.total_input_tokens * 0.00250 + self.total_output_tokens * 0.01000) / 1000
                    else:  # gemini
                        total_cost = (self.total_input_tokens * 0.000075 + self.total_output_tokens * 0.0003) / 1000
                    
                    api_name = "GPT-4o" if self.api_type == 'gpt' else "Gemini 2.0 Flash"
                    print(f"\n{self.colors.BLUE}📊 현재까지 총 사용량 ({api_name}){self.colors.ENDC}")
                    print(f"입력 토큰: {self.total_input_tokens:,}")
                    print(f"출력 토큰: {self.total_output_tokens:,}")
                    print(f"총 토큰: {total_tokens:,}")
                    cost_type = "실제" if self.api_type == 'gpt' else "추정"
                    print(f"{cost_type} 총 비용: ${total_cost:.6f}\n")
                    continue
                
                if not question.strip():
                    continue
                
                # API에 질문
                response = self.ask(question)
                api_display = "GPT-4o" if self.api_type == 'gpt' else "Gemini 2.0 Flash"
                print(f"\n{self.colors.GREEN}{api_display}:{self.colors.ENDC} {response}\n")
                
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
    parser = argparse.ArgumentParser(description="Seat Control Chatbot with OpenAI GPT-4o and Google Gemini")
    parser.add_argument("--prompt", type=str, 
                       default="SW/Chain2/main_v2/source/chain2_basic.txt",
                       help="Basic prompt file path")
    parser.add_argument("--sysprompt", type=str, 
                       default="SW/Chain2/main_v2/source/chain2_system.txt",
                       help="System prompt file path")
    parser.add_argument("--userprompt", type=str,
                       default="SW/Chain2/main_v2/source/chain2_option_list.json",
                       help="User prompt file path")
    args = parser.parse_args()
    
    # 챗봇 인스턴스 생성 및 실행
    chatbot = SeatControlChatbot()
    chatbot.run_chatbot(args.prompt, args.sysprompt, args.userprompt)


if __name__ == "__main__":
    main()