import json
import os
import time
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import google.generativeai as genai
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiOptionSelector:
    def __init__(self, 
                 secrets_path: str = "SW/tetris_secrets.json",
                 option_list_path: str = "SW/Chain2/main_v5/chain2_option_list.json",
                 system_prompt_path: str = "SW/Chain2/main_v5/base.txt",
                 input_text_path: str = "SW/Chain2/main_v5/scenarios/test.txt",
                 input_image_path: str = "SW/Chain2/main_v5/scenarios/test.jpg"):
        """
        Gemini API를 사용한 차량 좌석 레이아웃 선택기 초기화 (동기 방식)
        
        Args:
            secrets_path: API 키가 저장된 JSON 파일 경로
            option_list_path: 선택할 옵션들이 저장된 JSON 파일 경로
            system_prompt_path: 시스템 프롬프트가 저장된 텍스트 파일 경로
            input_text_path: API 입력용 텍스트 파일 경로 (시나리오 수행 자료)
            input_image_path: API 입력용 이미지 파일 경로 (시나리오 수행 자료)
        """
        self.secrets_path = secrets_path
        self.option_list_path = option_list_path
        self.system_prompt_path = system_prompt_path
        self.input_text_path = input_text_path
        self.input_image_path = input_image_path
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # 초기화 수행
        self._load_api_key()
        self._initialize_model()
    
    def _load_api_key(self) -> None:
        """API 키 로드 및 설정"""
        try:
            if not os.path.exists(self.secrets_path):
                raise FileNotFoundError(f"API 키 파일을 찾을 수 없습니다: {self.secrets_path}")
            
            with open(self.secrets_path, 'r', encoding='utf-8') as f:
                secrets = json.load(f)
            
            # Google API 키 추출
            api_key = None
            
            # "google" 섹션에서 API 키 찾기
            if "google" in secrets:
                google_section = secrets["google"]
                if isinstance(google_section, dict) and "GOOGLE_API_KEY" in google_section:
                    api_key = google_section["GOOGLE_API_KEY"]
                else:
                    raise ValueError("google 섹션에 GOOGLE_API_KEY가 없습니다.")
            else:
                raise ValueError("secrets 파일에 'google' 섹션을 찾을 수 없습니다.")
            
            if not api_key or api_key == "example":
                raise ValueError("유효한 Google API 키가 설정되지 않았습니다. 'example'을 실제 API 키로 교체해주세요.")
            
            genai.configure(api_key=api_key)
            print("🔑 Google Gemini 2.5 Flash API 키 로드 및 모델 초기화 완료")
            
        except Exception as e:
            print(f"🚨 API 키 로드 실패: {e}")
            raise
    
    def _initialize_model(self) -> None:
        """Gemini 모델 초기화"""
        try:
            # Gemini 2.5 Flash 모델 사용
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print(f"📋 사용 중인 모델: {self.model.model_name}")
            
        except Exception as e:
            print(f"🚨 모델 초기화 실패: {e}")
            raise
    
    def _load_file_content(self, file_path: str, is_json: bool = False) -> Any:
        """파일 내용 로드"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if is_json:
                    return json.load(f)
                else:
                    return f.read()
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류 ({file_path}): {e}")
            raise
        except Exception as e:
            logger.error(f"파일 로드 오류 ({file_path}): {e}")
            raise
    
    def load_system_prompt(self) -> str:
        """시스템 프롬프트 로드"""
        return self._load_file_content(self.system_prompt_path, is_json=False)
    
    def load_option_list(self) -> Dict[str, Any]:
        """옵션 리스트 로드"""
        return self._load_file_content(self.option_list_path, is_json=True)
    
    def load_scenario_content(self) -> Dict[str, Any]:
        """시나리오 수행을 위한 입력 파일들 로드 (텍스트 및 이미지)"""
        content = {}
        
        # 시나리오 텍스트 파일 로드
        try:
            content['text'] = self._load_file_content(self.input_text_path, is_json=False)
            print(f"📋 시나리오 로드됨:")
            print(f"텍스트: {self.input_text_path}")
        except Exception as e:
            print(f"🚨 시나리오 텍스트 파일을 찾을 수 없습니다: {e}")
            content['text'] = ""
        
        # 시나리오 이미지 파일 로드
        try:
            import PIL.Image
            if os.path.exists(self.input_image_path):
                content['image'] = PIL.Image.open(self.input_image_path)
                print(f"✅ 시나리오 이미지 로드됨: {self.input_image_path}")
                print(f"이미지: {self.input_image_path}")
            else:
                print(f"🚨 시나리오 이미지 파일을 찾을 수 없습니다: {self.input_image_path}")
                content['image'] = None
        except Exception as e:
            print(f"🚨 시나리오 이미지 파일 로드 실패: {e}")
            content['image'] = None
            
        return content
    
    def _create_option_summary(self, options: Dict[str, Any]) -> str:
        """옵션 리스트를 요약된 형태로 변환 (부하 감소)"""
        try:
            option_list = options.get('option_list', [])
            summary_lines = []
            
            for people_group in option_list:
                people_count = people_group.get('people_count', 0)
                cases = people_group.get('cases', [])
                
                for case in cases:
                    luggage = case.get('luggage_amount', '')
                    option_no = case.get('option_no', 0)
                    summary_lines.append(f"옵션 {option_no}: {people_count}명, 짐 크기 {luggage}")
            
            return "\n".join(summary_lines)
        except Exception as e:
            # 파싱 실패 시 전체 JSON 반환
            return json.dumps(options, ensure_ascii=False, indent=2)
    
    def create_selection_prompt(self, options: Dict[str, Any], scenario_content: Dict[str, Any]) -> str:
        """선택을 위한 프롬프트 생성"""
        sysprompt = self.load_system_prompt()
        
        # 옵션 리스트를 간단한 텍스트로 포맷팅 (부하 감소)
        option_summary = self._create_option_summary(options)
        
        # 시나리오 텍스트 내용 추가
        scenario_text = scenario_content.get('text', '')
        
        # f-string 중괄호 문제 해결을 위해 템플릿 분리
        full_prompt = f"""시스템 프롬프트:
{sysprompt}

옵션 리스트:
{option_summary}

시나리오:
{scenario_text}

위의 시나리오에 따라 적절한 좌석 레이아웃을 선택해주세요.
응답은 반드시 지정된 Output 형식(JSON)에 맞춰 출력하고, 선택 이유를 포함해주세요.

응답 형식:"""
        
        # JSON 예시를 별도로 추가 (f-string 충돌 방지)
        json_format = """
{
  "cell_layout": {
    "1": { "rail": "X", "pos": "A", "face": "F", "mode": "chair" },
    "2": { "rail": "Y", "pos": "B", "face": "R", "mode": "chair" },
    "3": { "rail": "X", "pos": "C", "face": "B", "mode": "chair" },
    "4": { "rail": "Y", "pos": "A", "face": "L", "mode": "storage" }
  },
  "option_no": 숫자,
  "reason": "선택 이유"
}"""
        
        return full_prompt + json_format
    
    def _estimate_tokens(self, text):
        """토큰 수 추정 함수"""
        if not text:
            return 0
        # 한국어와 영어를 고려한 토큰 추정
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - korean_chars - len(''.join(re.findall(r'[a-zA-Z]+', text)))
        
        estimated_tokens = korean_chars + (english_words * 0.75) + (other_chars * 0.25)
        return int(estimated_tokens)
    
    def select_option(self, custom_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """동기 방식으로 차량 좌석 레이아웃 선택 (시나리오 자료 기반)"""
        try:
            # 옵션 로드 (커스텀 옵션이 제공되면 사용, 아니면 파일에서 로드)
            options = custom_options if custom_options else self.load_option_list()
            
            # 시나리오 수행을 위한 입력 컨텐츠 로드
            scenario_content = self.load_scenario_content()
            
            # 프롬프트 생성
            prompt = self.create_selection_prompt(options, scenario_content)
            
            # API 호출용 컨텐츠 준비
            content_parts = [prompt]
            if scenario_content.get('image'):
                content_parts.append(scenario_content['image'])
            
            print(f"\n🤖 Gemini 2.5 Flash 처리 중...")
            
            start_time = time.time()
            
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
            
            # Gemini API 호출 (동기 방식)
            try:
                response = self.model.generate_content(
                    content_parts,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # 응답 처리 개선 - finish_reason 체크
                if not response.candidates or len(response.candidates) == 0:
                    print("🚨 응답 후보가 없습니다.")
                    raise Exception("No candidates returned")
                
                candidate = response.candidates[0]
                
                # finish_reason이 SAFETY(2)인 경우 처리
                if candidate.finish_reason == 2:  # SAFETY
                    print("⚠️ 안전 필터에 의해 응답이 차단되었습니다. 더 완화된 설정으로 재시도합니다.")
                    
                    # 안전 설정을 더 완화하여 재시도
                    relaxed_safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
                    ]
                    
                    print("🔄 완화된 안전 설정으로 재시도 중...")
                    response = self.model.generate_content(
                        content_parts,
                        generation_config=generation_config,
                        safety_settings=relaxed_safety_settings
                    )
                    
                    # 재시도 후에도 차단되면 기본값 반환
                    if response.candidates and response.candidates[0].finish_reason == 2:
                        print("🚨 재시도 후에도 안전 필터에 차단되었습니다. 기본 레이아웃을 반환합니다.")
                        default_layout = {
                            "1": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                            "2": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                            "3": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                            "4": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"}
                        }
                        return {
                            'cell_layout': default_layout,
                            'option_no': 10,
                            'reason': '안전 필터로 인한 기본 선택 (4명, 짐 크기 S)',
                            'raw_response': "응답이 안전 필터에 의해 차단됨"
                        }
                
                # content가 없는 경우 처리
                if not candidate.content or not candidate.content.parts:
                    print("🚨 응답 내용이 비어있습니다. 기본 레이아웃을 반환합니다.")
                    default_layout = {
                        "1": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                        "2": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                        "3": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                        "4": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"}
                    }
                    return {
                        'cell_layout': default_layout,
                        'option_no': 10,
                        'reason': '빈 응답으로 인한 기본 선택 (4명, 짐 크기 S)',
                        'raw_response': "응답 내용이 비어있음"
                    }
                
                # 정상 응답 처리
                content = response.text
                
            except Exception as api_error:
                print(f"🚨 API 호출 중 오류 발생: {api_error}")
                # API 오류 시 기본 레이아웃 반환
                default_layout = {
                    "1": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                    "2": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                    "3": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                    "4": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"}
                }
                return {
                    'cell_layout': default_layout,
                    'option_no': 10,
                    'reason': f'API 오류로 인한 기본 선택: {str(api_error)}',
                    'raw_response': f"API 오류: {str(api_error)}"
                }
            
            # 토큰 사용량 계산
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
            else:
                input_tokens = self._estimate_tokens(prompt)
                output_tokens = self._estimate_tokens(content)
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # 토큰 정보 출력
            print(f"📊 토큰: {input_tokens}→{output_tokens} | 시간: {response_time:.2f}s | 모델: Gemini 2.5 Flash")
            
            # 응답 출력
            print(f"✅ Gemini 2.5 Flash 응답:")
            print("-" * 30)
            print(content)
            print("-" * 30)
            
            # 응답 파싱
            result = self._parse_response(content, options)
            
            # 총 토큰 사용량
            total_tokens = self.total_input_tokens + self.total_output_tokens
            print(f"\n📊 총 사용 토큰: {total_tokens:,}")
            
            return result
            
        except Exception as e:
            print(f"🚨 옵션 선택 오류: {e}")
            raise
    
    def _parse_response(self, response_text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Gemini 응답 파싱 - 차량 좌석 레이아웃 전용"""
        try:
            # JSON 부분 추출 시도
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # 필수 필드 확인
                if 'cell_layout' in parsed and 'option_no' in parsed:
                    return {
                        'cell_layout': parsed['cell_layout'],
                        'option_no': parsed['option_no'],
                        'reason': parsed.get('reason', ''),
                        'raw_response': response_text
                    }
            
            # JSON 파싱 실패 시 기본값 반환
            print("⚠️ 응답 파싱 실패, 기본 옵션을 선택합니다.")
            default_layout = {
                "1": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                "2": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                "3": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                "4": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"}
            }
            return {
                'cell_layout': default_layout,
                'option_no': 10,
                'reason': '응답 파싱 실패로 인한 기본 선택 (4명, 짐 크기 S)',
                'raw_response': response_text
            }
            
        except Exception as e:
            print(f"🚨 응답 파싱 오류: {e}")
            # 에러 시 기본 레이아웃 반환
            default_layout = {
                "1": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                "2": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                "3": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                "4": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"}
            }
            return {
                'cell_layout': default_layout,
                'option_no': 10,
                'reason': f'파싱 오류: {str(e)}',
                'raw_response': response_text
            }

def main():
    """사용 예시 - 차량 좌석 레이아웃 선택 시스템"""
    try:
        # 옵션 선택기 초기화
        selector = GeminiOptionSelector()
        
        # 시나리오 파일들(test.txt, test.jpg)을 기반으로 레이아웃 선택 실행
        result = selector.select_option()
        
        # 결과 출력 (차량 좌석 레이아웃 전용)
        print("=== 차량 좌석 레이아웃 선택 결과 ===")
        print(f"옵션 번호: {result['option_no']}")
        print(f"선택 이유: {result['reason']}")
        print("좌석 배치:")
        for seat_id, config in result['cell_layout'].items():
            print(f"  좌석 {seat_id}: {config}")
        
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()