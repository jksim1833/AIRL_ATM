from pathlib import Path
import os
import json
import re
import argparse
import google.generativeai as genai
import time
from PIL import Image

class SeatControlChatbot:
    def __init__(self):
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    def load_api_keys(self):
        """Gemini API 키 로드 및 클라이언트 초기화"""
        possible_paths = [
            Path.home() / "Desktop" / "AIRL_ATM" / "SW" / "tetris_secrets.json",
            Path.home() / "Desktop" / "AIRL_ATM" / "tetris_secrets.json",
            Path("C:/Users/dongy/Desktop/임베디드/AIRL_ATM/SW/tetris_secrets.json"),
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
                print(f"   - {path}")
            raise FileNotFoundError("🚨 tetris_secrets.json 파일을 찾을 수 없습니다.")
        
        try:
            with open(secrets_path, 'r', encoding='utf-8') as f:
                secrets = json.load(f)
            
            self._initialize_gemini(secrets)
                
        except KeyError as e:
            raise KeyError(f"🚨 API 키를 찾을 수 없습니다: {e}")
        except json.JSONDecodeError:
            raise ValueError("🚨 tetris_secrets.json 파일 형식이 올바르지 않습니다")

    def _initialize_gemini(self, secrets):
        """Google Gemini 2.5 Flash 초기화"""
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
        
        try:
            # Gemini 2.5 Flash 모델 초기화 (정확한 모델명 사용)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("🔑 Google Gemini 2.5 Flash API 키 로드 및 모델 초기화 완료")
            
            # 모델 정보 확인
            print(f"📋 사용 중인 모델: {self.model.model_name}")
            
        except Exception as e:
            print(f"⚠️ Gemini 2.5 Flash 초기화 실패, 대체 모델 사용 시도: {e}")
            try:
                # 대체 모델들 시도
                alternative_models = [
                    'gemini-2.0-flash-exp',
                    'gemini-1.5-flash',
                    'gemini-1.5-pro'
                ]
                
                for model_name in alternative_models:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        print(f"🔄 대체 모델 사용: {model_name}")
                        break
                    except:
                        continue
                        
                if not self.model:
                    raise Exception("사용 가능한 Gemini 모델이 없습니다")
                    
            except Exception as fallback_error:
                raise Exception(f"Gemini 모델 초기화 실패: {fallback_error}")

    def load_prompts(self, sysprompt_path, basic_prompt_path, option_list_path):
        """프롬프트 파일들 로드"""
        try:
            # 시스템 프롬프트 로드
            with open(sysprompt_path, "r", encoding='utf-8') as f:
                sysprompt = f.read()
            
            # 기본 프롬프트 로드
            with open(basic_prompt_path, "r", encoding='utf-8') as f:
                basic_prompt = f.read()
                
            # 옵션 리스트 로드
            with open(option_list_path, "r", encoding='utf-8') as f:
                option_list = f.read()
                
            return sysprompt, basic_prompt, option_list
        except FileNotFoundError as e:
            print(f"🚨 프롬프트 파일을 찾을 수 없습니다: {e}")
            # 기본 프롬프트 사용
            sysprompt = "You are a helpful seat control assistant. Provide seat control commands in JSON format."
            basic_prompt = "Initialize seat control system"
            option_list = "{}"
            return sysprompt, basic_prompt, option_list

    def load_scenario(self, scenario_txt_path, scenario_img_path=None):
        """시나리오 파일 로드 (텍스트 + 이미지)"""
        try:
            # 텍스트 시나리오 로드
            with open(scenario_txt_path, "r", encoding='utf-8') as f:
                scenario_text = f.read()
            
            # 이미지 로드 (있는 경우)
            scenario_image = None
            if scenario_img_path and os.path.exists(scenario_img_path):
                scenario_image = Image.open(scenario_img_path)
                print(f"✅ 시나리오 이미지 로드됨: {scenario_img_path}")
            
            return scenario_text, scenario_image
        except FileNotFoundError as e:
            print(f"🚨 시나리오 파일을 찾을 수 없습니다: {e}")
            return "기본 시나리오", None

    def extract_python_code(self, content):
        """응답에서 Python 코드 추출"""
        code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
        code_blocks = code_block_regex.findall(content)
        if code_blocks:
            full_code = "\n".join(code_blocks)
            if full_code.startswith("python"):
                full_code = full_code[7:]
            return full_code
        return None

    def ask(self, prompt, image=None):
        """Gemini 2.5 Flash에게 질문하고 응답 받기"""
        start_time = time.time()
        
        # 프롬프트 구성
        content_parts = [prompt]
        if image:
            content_parts.append(image)

        # Gemini 2.5 Flash 최적화된 설정으로 생성
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=2000,
            temperature=0.7,
            top_p=0.8,
            top_k=40
        )
        
        # 안전 설정
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        
        try:
            # API 호출
            response = self.model.generate_content(
                content_parts,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # 응답에 유효한 후보가 있는지 확인
            if not response.candidates:
                # 응답에 후보가 없으므로 오류 처리
                finish_reason = response.prompt_feedback.finish_reason.name if response.prompt_feedback.finish_reason else "UNKNOWN"
                error_message = f"🚨 Gemini 2.5 Flash API 호출 중 오류 발생: 유효한 응답을 받지 못했습니다. (finish_reason: {finish_reason})"
                print(error_message)
                return "죄송합니다. 요청을 처리할 수 없습니다."
            
            content = response.text
            
            # 토큰 사용량 계산
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                # 실제 토큰 사용량이 있다면 사용
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
            else:
                # 추정값 사용
                input_tokens = self._estimate_tokens(prompt)
                output_tokens = self._estimate_tokens(content)
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # 간소화된 토큰 정보 출력
            print(f"📊 토큰: {input_tokens}→{output_tokens} | 시간: {response_time:.2f}s | 모델: Gemini 2.5 Flash")
            
            return content
            
        except Exception as e:
            print(f"🚨 Gemini 2.5 Flash API 호출 중 예외 발생: {e}")
            if hasattr(e, 'message'):
                print(f"상세 오류: {e.message}")
            return "죄송합니다. 응답을 생성할 수 없습니다."

    def _estimate_tokens(self, text):
        """토큰 수 추정 함수"""
        if not text:
            return 0
        # 한국어와 영어를 고려한 토큰 추정
        # 일반적으로 한국어 1글자 = 1토큰, 영어 4글자 = 1토큰 정도
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - korean_chars - len(''.join(re.findall(r'[a-zA-Z]+', text)))
        
        estimated_tokens = korean_chars + (english_words * 0.75) + (other_chars * 0.25)
        return int(estimated_tokens)

    def run_scenario_test(self, scenario_txt_path, scenario_img_path,
                          sysprompt_path="SW/Chain2/main_v3/source/chain2_system.txt",
                          basic_prompt_path="SW/Chain2/main_v3/source/chain2_basic.txt",
                          option_list_path="SW/Chain2/main_v3/source/chain2_option_list.json"):
        """시나리오 테스트 실행"""
        try:
            print("=" * 50)
            print("🚀 Seat Control Scenario Test")
            print("=" * 50)
            
            # Gemini API 키 로드
            print("Initializing Gemini 2.5 Flash...")
            self.load_api_keys()
            
            # 프롬프트 로드
            sysprompt, basic_prompt, option_list = self.load_prompts(
                sysprompt_path, basic_prompt_path, option_list_path
            )
            
            # 시나리오 로드
            scenario_text, scenario_image = self.load_scenario(scenario_txt_path, scenario_img_path)
            
            print(f"\n📋 시나리오 로드됨:")
            print(f"텍스트: {scenario_txt_path}")
            if scenario_image:
                print(f"이미지: {scenario_img_path}")
            
            # 전체 프롬프트 구성
            full_prompt = f"""시스템 프롬프트:
{sysprompt}

기본 프롬프트:
{basic_prompt}

옵션 리스트:
{option_list}

시나리오:
{scenario_text}

위의 시나리오에 따라 적절한 좌석 제어 명령을 생성해주세요. 
응답에는 반드시 option_list에서 선택한 옵션 번호를 명시해주세요.
예: "옵션 3번 선택: ..." 형태로 답변해주세요."""
            
            print(f"\n🤖 Gemini 2.5 Flash 처리 중...")
            
            # AI에 질문
            response = self.ask(full_prompt, scenario_image)
            
            print(f"\n✅ Gemini 2.5 Flash 응답:")
            print("-" * 30)
            print(response)
            print("-" * 30)
            
            # Python 코드 추출 및 실행
            code = self.extract_python_code(response)
            if code is not None:
                print("\n🔧 좌석 제어 코드 실행 중...")
                try:
                    exec(code)
                    print("✅ 좌석 제어 실행 완료!")
                except Exception as e:
                    print(f"❌ 좌석 제어 실행 오류: {e}")
            else:
                print("ℹ️ 실행 가능한 코드가 응답에 포함되지 않았습니다.")
            
            # 최종 토큰 사용량
            total_tokens = self.total_input_tokens + self.total_output_tokens
            print(f"\n📊 총 사용 토큰: {total_tokens:,}")
            
        except Exception as e:
            print(f"🚨 시나리오 테스트 실행 중 오류 발생: {e}")

    def run_chatbot(self, sysprompt_path="SW/Chain2/main_v3/source/chain2_system.txt",
                      basic_prompt_path="SW/Chain2/main_v3/source/chain2_basic.txt",
                      option_list_path="SW/Chain2/main_v3/source/chain2_option_list.json"):
        """챗봇 모드"""
        try:
            print("=" * 50)
            print("🚀 Seat Control Chatbot")
            print("=" * 50)
            
            # Gemini API 키 로드
            print("Initializing Gemini 2.5 Flash...")
            self.load_api_keys()
            
            # 프롬프트 로드
            sysprompt, basic_prompt, option_list = self.load_prompts(
                sysprompt_path, basic_prompt_path, option_list_path
            )
            
            print("Done.")
            print("Welcome to the Seat Control chatbot using Gemini 2.5 Flash!")
            print("Commands: !quit or !exit to exit, !clear to clear screen")
            
            # 메인 챗봇 루프
            while True:
                question = input("SeatControl> ")
                
                if question in ["!quit", "!exit"]:
                    total_tokens = self.total_input_tokens + self.total_output_tokens
                    print(f"\n🔚 세션 종료 - 총 사용 토큰: {total_tokens:,}")
                    break
                
                if question == "!clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue
                
                if not question.strip():
                    continue
                
                # 전체 프롬프트 구성
                full_prompt = f"""시스템 프롬프트:
{sysprompt}

기본 프롬프트:
{basic_prompt}

옵션 리스트:
{option_list}

사용자 요청: {question}

위의 요청에 따라 적절한 좌석 제어 명령을 생성해주세요.
응답에는 반드시 option_list에서 선택한 옵션 번호를 명시해주세요."""
                
                # Gemini에 질문
                response = self.ask(full_prompt)
                print(f"\nGemini 2.5 Flash: {response}\n")
                
                # Python 코드 추출 및 실행
                code = self.extract_python_code(response)
                if code is not None:
                    print("좌석 제어 명령 실행 중...")
                    try:
                        exec(code)
                        print("✅ 좌석 제어 실행 완료!\n")
                    except Exception as e:
                        print(f"❌ 좌석 제어 실행 오류: {e}\n")
                        
        except Exception as e:
            print(f"🚨 챗봇 실행 중 오류 발생: {e}")


def main():
    """메인 함수 - 기본적으로 시나리오 테스트 모드로 실행"""
    parser = argparse.ArgumentParser(description="Seat Control Chatbot - Gemini Only")
    parser.add_argument("--mode", type=str, choices=['chat', 'scenario'], default='scenario',
                        help="실행 모드 선택: chat (챗봇 모드) 또는 scenario (시나리오 테스트 모드)")
    parser.add_argument("--scenario-txt", type=str, 
                        default="SW/Chain2/main_v3/scenarios/test2.txt",
                        help="시나리오 텍스트 파일 경로")
    parser.add_argument("--scenario-img", type=str,
                        default="SW/Chain2/main_v3/scenarios/test2.jpg",
                        help="시나리오 이미지 파일 경로")
    parser.add_argument("--sysprompt", type=str, 
                        default="SW/Chain2/main_v3/source/chain2_system.txt",
                        help="시스템 프롬프트 파일 경로")
    parser.add_argument("--basic-prompt", type=str,
                        default="SW/Chain2/main_v3/source/chain2_basic.txt", 
                        help="기본 프롬프트 파일 경로")
    parser.add_argument("--option-list", type=str,
                        default="SW/Chain2/main_v3/source/chain2_option_list.json",
                        help="옵션 리스트 파일 경로")
    args = parser.parse_args()
    
    # 챗봇 인스턴스 생성
    chatbot = SeatControlChatbot()
    
    if args.mode == 'scenario':
        # 시나리오 테스트 모드 (기본값)
        chatbot.run_scenario_test(
            args.scenario_txt, args.scenario_img,
            args.sysprompt, args.basic_prompt, args.option_list
        )
    else:
        # 챗봇 모드
        chatbot.run_chatbot(args.sysprompt, args.basic_prompt, args.option_list)


if __name__ == "__main__":
    main()