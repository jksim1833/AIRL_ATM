#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최적화된 Gemini 차량 좌석 레이아웃 선택 시스템 v2.1
- main() 함수 중복 선언 오류 수정
- 강화된 JSON 파싱 및 복구 로직
- API 응답 토큰 사용량 처리 로직 개선
- 전반적인 가독성 및 안정성 향상
"""

import json
import os
import time
import re
from typing import Dict, Any, Optional, List
import google.generativeai as genai
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedGeminiSelector:
    """
    Google Gemini API를 사용하여 차량 좌석 레이아웃을 선택하는 클래스.
    파일 로드, API 통신, 응답 파싱 등 모든 과정에 대해 강화된 오류 처리가 적용되어 있습니다.
    """
    def __init__(self, 
                 secrets_path: str = "SW/tetris_secrets.json",
                 option_list_path: str = "SW/Chain2/main_v5/chain2_option_list.json",
                 system_prompt_path: str = "SW/Chain2/main_v5/base.txt",
                 input_text_path: str = "SW/Chain2/main_v5/scenarios/test.txt",
                 input_image_path: str = "SW/Chain2/main_v5/scenarios/test.jpg",
                 use_images: bool = False):
        """
        최적화된 Gemini API 차량 좌석 레이아웃 선택기 초기화.
        
        Args:
            secrets_path: API 키 파일 경로.
            option_list_path: 옵션 리스트 JSON 파일 경로.  
            system_prompt_path: 시스템 프롬프트 텍스트 파일 경로.
            input_text_path: 시나리오 텍스트 파일 경로.
            input_image_path: 시나리오 이미지 파일 경로 (선택사항).
            use_images: 이미지 사용 여부 (기본: False, 안전 필터 방지).
        """
        self.secrets_path = secrets_path
        self.option_list_path = option_list_path
        self.system_prompt_path = system_prompt_path
        self.input_text_path = input_text_path
        self.input_image_path = input_image_path
        self.use_images = use_images
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # 초기화 수행
        self._load_api_key()
        self._initialize_model()
    
    def _load_api_key(self) -> None:
        """API 키 로드 및 설정 (다중 경로 지원)"""
        possible_paths = [
            Path(self.secrets_path),
            Path("SW/tetris_secrets.json"),
            Path("tetris_secrets.json"),
            Path.home() / "Desktop" / "AIRL_ATM" / "SW" / "tetris_secrets.json",
            Path.cwd() / "SW" / "tetris_secrets.json",
        ]
        
        secrets_data = None
        used_path = None
        
        for path in possible_paths:
            try:
                if path.exists() and path.is_file():
                    with open(path, 'r', encoding='utf-8') as f:
                        secrets_data = json.load(f)
                    used_path = path
                    print(f"API 키 파일 발견: {path}")
                    break
            except (json.JSONDecodeError, PermissionError, UnicodeDecodeError) as e:
                logger.warning(f"경로 {path}에서 파일 로드 실패: {e}")
                continue
            except Exception as e:
                logger.warning(f"예상치 못한 오류 ({path}): {e}")
                continue
        
        if not secrets_data:
            raise FileNotFoundError(
                f"API 키 파일을 찾을 수 없습니다. 다음 위치를 확인해주세요:\n" +
                "\n".join([f"  - {p}" for p in possible_paths])
            )
        
        # API 키 추출 및 검증
        try:
            api_key = None
            
            # 다양한 키 구조 지원
            if isinstance(secrets_data, dict):
                if "google" in secrets_data and isinstance(secrets_data["google"], dict):
                    api_key = secrets_data["google"].get("GOOGLE_API_KEY")
                elif "GOOGLE_API_KEY" in secrets_data:
                    api_key = secrets_data["GOOGLE_API_KEY"]
                elif "google_api_key" in secrets_data:
                    api_key = secrets_data["google_api_key"]
            
            if not api_key:
                raise KeyError("Google API 키를 찾을 수 없습니다. 'google.GOOGLE_API_KEY' 또는 'GOOGLE_API_KEY' 필드가 필요합니다.")
            
            if not isinstance(api_key, str) or len(api_key.strip()) == 0:
                raise ValueError("API 키가 유효하지 않습니다 (빈 문자열)")
            
            if api_key.strip().lower() in ["example", "your_api_key_here", "placeholder"]:
                raise ValueError("API 키를 실제 값으로 교체해주세요")
            
            genai.configure(api_key=api_key.strip())
            print(f"Google Gemini API 키 로드 완료 (경로: {used_path})")
            
        except Exception as e:
            raise Exception(f"API 키 설정 실패: {e}")
    
    def _initialize_model(self) -> None:
        """Gemini 모델 초기화 (대체 모델 지원)"""
        models_to_try = [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro',
            'gemini-2.0-flash-exp',
        ]
        
        for model_name in models_to_try:
            try:
                self.model = genai.GenerativeModel(model_name)
                print(f"사용 중인 모델: {self.model.model_name}")
                return
            except Exception as e:
                logger.warning(f"모델 {model_name} 초기화 실패: {e}")
                continue
        
        raise Exception("사용 가능한 Gemini 모델이 없습니다. API 키와 네트워크 연결을 확인해주세요.")
    
    def _load_file_safely(self, file_path: str, is_json: bool = False, default_content: Any = None) -> Any:
        """안전한 파일 로드 (강화된 예외 처리)"""
        try:
            path = Path(file_path)
            if not path.exists():
                if default_content is not None:
                    logger.info(f"파일 없음 ({file_path}), 기본값 사용")
                    return default_content
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            if not path.is_file():
                raise ValueError(f"경로가 파일이 아닙니다: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                if is_json:
                    content = json.load(f)
                    # JSON 기본 구조 검증
                    if not isinstance(content, dict):
                        raise ValueError("JSON 파일의 최상위 구조가 객체가 아닙니다")
                    return content
                else:
                    content = f.read().strip()
                    if len(content) == 0:
                        logger.warning(f"파일이 비어있습니다: {file_path}")
                    return content
                    
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"파일 형식 오류 ({file_path}): {e}")
            if default_content is not None:
                return default_content
            raise
        except Exception as e:
            logger.error(f"파일 로드 오류 ({file_path}): {e}")
            if default_content is not None:
                return default_content
            raise
    
    def load_system_prompt(self) -> str:
        """시스템 프롬프트 로드 (기본값 포함)"""
        default_prompt = """You are an assistant that helps select optimal vehicle seat layouts.
Analyze the scenario and select the best option from the provided list.
Output in JSON format with cell_layout, option_no, and reason."""
        
        return self._load_file_safely(self.system_prompt_path, is_json=False, default_content=default_prompt)
    
    def load_option_list(self) -> Dict[str, Any]:
        """옵션 리스트 로드 (기본값 포함)"""
        default_options = {
            "option_list": [
                {
                    "people_count": 4,
                    "cases": [
                        {
                            "luggage_amount": "S",
                            "cell_layout": {
                                "1": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                                "2": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                                "3": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
                                "4": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"}
                            },
                            "option_no": 10
                        }
                    ]
                }
            ]
        }
        
        options = self._load_file_safely(self.option_list_path, is_json=True, default_content=default_options)
        
        # 옵션 리스트 구조 검증
        if "option_list" not in options or not isinstance(options["option_list"], list):
            logger.warning("옵션 리스트 구조가 올바르지 않습니다. 기본값을 사용합니다.")
            return default_options
            
        return options
    
    def load_scenario_content(self) -> Dict[str, Any]:
        """시나리오 콘텐츠 로드 (텍스트 + 선택적 이미지)"""
        content = {}
        
        # 텍스트 로드
        default_text = "4명의 승객이 여행을 위해 중간 크기의 가방들을 가지고 있습니다."
        content['text'] = self._load_file_safely(self.input_text_path, is_json=False, default_content=default_text)
        
        if self.input_text_path and Path(self.input_text_path).exists():
            print(f"시나리오 텍스트 로드됨: {self.input_text_path}")
        else:
            print("기본 시나리오 텍스트 사용")
        
        # 이미지 로드 (선택적)
        content['image'] = None
        if self.use_images:
            try:
                import PIL.Image
                image_path = Path(self.input_image_path)
                if image_path.exists() and image_path.is_file():
                    content['image'] = PIL.Image.open(image_path)
                    print(f"시나리오 이미지 로드됨: {self.input_image_path}")
                else:
                    print(f"이미지 파일 없음: {self.input_image_path}")
            except ImportError:
                print("PIL 라이브러리가 없습니다. 이미지 기능을 비활성화합니다.")
            except Exception as e:
                print(f"이미지 로드 실패: {e}")
        else:
            print("이미지 사용 비활성화 (안전 필터 방지)")
            
        return content
    
    def _create_option_summary(self, options: Dict[str, Any]) -> str:
        """옵션 리스트를 간단한 요약으로 변환 (토큰 절약)"""
        try:
            option_list = options.get('option_list', [])
            summary_lines = []
            
            for people_group in option_list:
                people_count = people_group.get('people_count', 0)
                cases = people_group.get('cases', [])
                
                for case in cases:
                    luggage = case.get('luggage_amount', '')
                    option_no = case.get('option_no', 0)
                    summary_lines.append(f"Option {option_no}: {people_count} people, luggage size {luggage}")
            
            return "\n".join(summary_lines[:12])
        except Exception as e:
            logger.warning(f"옵션 요약 생성 실패: {e}")
            return "Available options: 1-12 for different passenger and luggage combinations"
    
    def create_optimized_prompt(self, options: Dict[str, Any], scenario_content: Dict[str, Any]) -> str:
        """최적화된 프롬프트 생성 (안전 필터 및 토큰 절약)"""
        sysprompt = self.load_system_prompt()
        option_summary = self._create_option_summary(options)
        scenario_text = scenario_content.get('text', '')
        
        # 안전한 프롬프트 구성 (f-string 중괄호 충돌 방지)
        prompt_template = """System Instructions:
{sysprompt}

Available Options:
{option_summary}

Scenario:
{scenario_text}

Please select the most appropriate seat layout option and respond in this exact JSON format:
{{
  "cell_layout": {{
    "1": {{ "rail": "X", "pos": "B", "face": "F", "mode": "chair" }},
    "2": {{ "rail": "X", "pos": "B", "face": "F", "mode": "chair" }},
    "3": {{ "rail": "X", "pos": "B", "face": "F", "mode": "chair" }},
    "4": {{ "rail": "X", "pos": "B", "face": "F", "mode": "chair" }}
  }},
  "option_no": 10,
  "reason": "Brief explanation"
}}"""
        
        return prompt_template.format(
            sysprompt=sysprompt,
            option_summary=option_summary,
            scenario_text=scenario_text
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """개선된 토큰 수 추정"""
        if not text or not isinstance(text, str):
            return 0
            
        # 더 정확한 토큰 추정 로직
        # 한글: 1자 = 1토큰, 영어: 평균 4자 = 1토큰, 특수문자/공백: 평균 2자 = 1토큰
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        numbers = len(re.findall(r'\d', text))
        punctuation = len(re.findall(r'[^\w\s가-힣]', text))
        spaces = len(re.findall(r'\s', text))
        
        estimated = (
            korean_chars * 1.0 +         # 한글
            english_chars * 0.25 +       # 영어
            numbers * 0.5 +              # 숫자
            punctuation * 0.5 +          # 특수문자
            spaces * 0.1                 # 공백
        )
        
        return max(1, int(estimated))
    
    def select_layout(self, custom_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """차량 좌석 레이아웃 선택 (최적화된 버전)"""
        try:
            # 옵션 및 시나리오 로드
            options = custom_options if custom_options else self.load_option_list()
            scenario_content = self.load_scenario_content()
            
            # 최적화된 프롬프트 생성
            prompt = self.create_optimized_prompt(options, scenario_content)
            
            # API 호출 준비
            content_parts = [prompt]
            if self.use_images and scenario_content.get('image'):
                content_parts.append(scenario_content['image'])
            
            print("Gemini API 처리 중...")
            start_time = time.time()
            
            # 최적화된 안전 설정 (순서 개선)
            safety_configurations = [
                # 1단계: 최소 제한 (대부분 통과)
                [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ],
                # 2단계: 완화된 설정
                [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
                ],
                # 3단계: 기본 설정 (fallback)
                [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                ]
            ]
            
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=800,  # 더 절약
                temperature=0.2,        # 더 일관적
                top_p=0.8,
                top_k=40
            )
            
            response = None
            last_error = None
            
            # 안전 설정 단계별 시도
            for i, safety_settings in enumerate(safety_configurations):
                try:
                    if i > 0:
                        print(f"안전 설정 레벨 {i+1}로 재시도...")
                    
                    response = self.model.generate_content(
                        content_parts,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    # 응답 유효성 검사 (더 엄격)
                    if (response and 
                        hasattr(response, 'candidates') and response.candidates and 
                        len(response.candidates) > 0 and 
                        hasattr(response.candidates[0], 'content') and response.candidates[0].content and
                        hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts and
                        len(response.candidates[0].content.parts) > 0):
                        
                        # finish_reason 확인
                        finish_reason = response.candidates[0].finish_reason
                        if finish_reason == 1:  # STOP (정상 완료)
                            break
                        elif finish_reason == 2:  # SAFETY (안전 필터)
                            if i < len(safety_configurations) - 1:
                                continue  # 다음 설정 시도
                            else:
                                raise Exception("모든 안전 설정에서 차단됨")
                        else:
                            print(f"경고: finish_reason = {finish_reason}")
                            break # 일단 진행
                        
                except Exception as e:
                    last_error = e
                    logger.warning(f"안전 설정 레벨 {i+1} 실패: {e}")
                    if i == len(safety_configurations) - 1:  # 마지막 시도
                        raise Exception(f"모든 시도 실패. 마지막 오류: {last_error}")
            
            if not response or not hasattr(response, 'candidates') or not response.candidates:
                raise Exception(f"API 응답 없음. 마지막 오류: {last_error}")
            
            # 응답 처리
            content_text = response.text
            
            if not content_text or len(content_text.strip()) == 0:
                raise Exception("빈 응답 수신")
            
            # 토큰 계산 (실제 토큰 수 우선 사용)
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(content_text)
            
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                try:
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                except AttributeError:
                    pass  # 추정값 사용
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # 결과 출력
            print(f"토큰: {input_tokens}→{output_tokens} | 시간: {response_time:.2f}s")
            print("Gemini 응답:")
            print("-" * 30)
            print(content_text)
            print("-" * 30)
            
            # 응답 파싱
            result = self._parse_response(content_text, options)
            
            total_tokens = self.total_input_tokens + self.total_output_tokens
            print(f"총 사용 토큰: {total_tokens:,}")
            
            return result
            
        except Exception as e:
            print(f"레이아웃 선택 오류: {e}")
            return self._get_default_layout(f"오류 발생: {str(e)}")
    
    def _parse_response(self, response_text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """강화된 응답 파싱 (다중 패턴 지원)"""
        if not response_text:
            return self._get_default_layout("빈 응답")
        
        try:
            # 개선된 JSON 추출 패턴들
            # 복잡한 패턴보다 단순하고 순차적인 접근으로 안정성을 높임
            # 1. 코드 블록 내 JSON (가장 흔한 형식)
            code_block_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # 2. 코드 블록 없는 순수 JSON
                json_str = response_text
            
            # JSON 파싱
            parsed_data = json.loads(json_str)
            
            # 파싱 성공 시 결과 검증 및 반환
            if isinstance(parsed_data, dict):
                cell_layout = parsed_data.get('cell_layout')
                option_no = parsed_data.get('option_no')
                reason = parsed_data.get('reason')

                # 필수 필드 누락 시 기본값 반환
                if not cell_layout or not option_no or not reason:
                    return self._get_default_layout("필수 JSON 필드 누락")
                
                # 타입 검증 및 변환
                if not isinstance(cell_layout, dict):
                    cell_layout = self._get_default_layout("cell_layout 형식 오류")['cell_layout']
                if not isinstance(option_no, int):
                    try:
                        option_no = int(option_no)
                    except (ValueError, TypeError):
                        option_no = 10
                if not isinstance(reason, str):
                    reason = str(reason) if reason else "형식 오류 수정됨"
                
                print("JSON 파싱 성공")
                return {
                    'cell_layout': cell_layout,
                    'option_no': option_no,
                    'reason': reason,
                    'raw_response': response_text
                }
            
            # 파싱 실패 시 기본값
            print("JSON 파싱 실패, 기본 레이아웃 사용")
            return self._get_default_layout("JSON 파싱 실패")
        
        except json.JSONDecodeError as e:
            print(f"JSON 디코딩 오류: {e}")
            return self._get_default_layout(f"JSON 디코딩 오류: {str(e)}")
        except Exception as e:
            print(f"응답 파싱 오류: {e}")
            return self._get_default_layout(f"파싱 오류: {str(e)}")
    
    def _get_default_layout(self, reason: str) -> Dict[str, Any]:
        """안전한 기본 레이아웃 반환"""
        default_layout = {
            "1": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
            "2": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
            "3": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"},
            "4": {"rail": "X", "pos": "B", "face": "F", "mode": "chair"}
        }
        return {
            'cell_layout': default_layout,
            'option_no': 10,
            'reason': reason,
            'raw_response': f"기본값 사용: {reason}"
        }

def create_sample_files():
    """최적화된 샘플 파일들 생성"""
    try:
        # 1. 디렉토리 생성
        scenarios_dir = Path("SW/Chain2/main_v5/scenarios")
        scenarios_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 안전한 시나리오 텍스트
        safe_scenario = """Travel Planning Information:
- Number of passengers: 4 people
- Luggage items: 2 travel bags, 2 backpacks, sports equipment
- Trip duration: 3 days 2 nights
- Destination: Resort

Luggage Details:
Medium-sized travel carriers and personal backpacks.
The luggage can be efficiently accommodated by utilizing trunk space effectively."""

        # 3. 시나리오 파일 생성
        scenario_file = scenarios_dir / "test.txt"
        with open(scenario_file, "w", encoding="utf-8") as f:
            f.write(safe_scenario)
        print(f"안전한 시나리오 파일 생성: {scenario_file}")
        
        return True
        
    except Exception as e:
        print(f"샘플 파일 생성 실패: {e}")
        return False

def main():
    """최적화된 메인 함수"""
    try:
        print("=" * 60)
        print("최적화된 차량 좌석 레이아웃 선택 시스템 v2.1")
        print("=" * 60)
        
        # 샘플 파일 생성
        create_sample_files()
        
        # 선택기 초기화 (이미지 비활성화로 안전 필터 방지)
        selector = OptimizedGeminiSelector(use_images=False)
        
        # 레이아웃 선택 실행
        result = selector.select_layout()
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("최종 선택 결과")
        print("=" * 60)
        print(f"옵션 번호: {result['option_no']}")
        print(f"선택 이유: {result['reason']}")
        print("\n좌석 배치 상세:")
        
        for seat_id, config in result['cell_layout'].items():
            rail = config.get('rail', 'X')
            pos = config.get('pos', 'B')
            face = config.get('face', 'F')
            mode = config.get('mode', 'chair')
            print(f"  좌석 {seat_id}: Rail={rail}, Position={pos}, Face={face}, Mode={mode}")
        
        print(f"\n프로그램 완료 (총 토큰: {selector.total_input_tokens + selector.total_output_tokens:,})")
        
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n프로그램 실행 오류: {e}")
        print("\n문제 해결 체크리스트:")
        print("1. tetris_secrets.json 파일의 API 키 설정 확인")
        print("2. 인터넷 연결 상태 확인")
        print("3. 파일 경로 및 읽기 권한 확인")
        print("4. Google AI API 할당량 확인")

if __name__ == "__main__":
    main()