#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최적화된 Gemini 차량 좌석 레이아웃 선택 시스템 v2.5
- 입력 JSON 데이터를 자연어 텍스트로 변환하여 모델 안정성 향상
- 기본 모델을 'gemini-2.5-flash'로 변경
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
    오류 발생 시 기본값 반환 대신 명확한 Exception을 발생시킵니다.
    """
    def __init__(self, 
                 secrets_path: str = "SW/tetris_secrets.json",
                 option_list_path: str = "SW/Chain2/main_v5/chain2_option_list.json",
                 system_prompt_path: str = "SW/Chain2/main_v5/base.txt",
                 input_text_path: str = "SW/Chain2/main_v5/scenarios/test.txt",
                 input_image_path: str = "",
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
        
        try:
            api_key = None
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
            'gemini-2.5-flash',
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
        """안전한 파일 로드 (강화된 예외 처리 및 기본값 반환)"""
        path = Path(file_path)
        
        if not file_path or not path.exists() or not path.is_file():
            if default_content is not None:
                logger.warning(f"파일을 찾을 수 없습니다 ({file_path}), 기본값 사용")
                return default_content
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if is_json:
                    content = json.load(f)
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
        
        content['image'] = None
        if self.use_images and self.input_image_path:
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
            print("이미지 사용 비활성화")
            
        return content

    def _convert_json_to_text(self, json_string: str) -> str:
        """JSON 문자열을 모델 친화적인 자연어 텍스트로 변환"""
        try:
            data = json.loads(json_string)
            people = data.get('people', 0)
            luggage_details = data.get('luggage_details', {})
            
            luggage_list = []
            for item in luggage_details.values():
                obj_name = item.get('object', '알 수 없는 짐')
                note = item.get('special_note', '').replace('공간 차지', '공간을 차지하는')
                
                if '킥보드' in obj_name and note:
                    luggage_list.append(f"{obj_name} ({note})")
                elif '헬멧' in obj_name and note:
                    luggage_list.append(f"{obj_name} ({note})")
                else:
                    luggage_list.append(obj_name)
                    
            luggage_text = ", ".join(luggage_list)
            
            return f"승객 {people}명과 총 {len(luggage_details)}개의 짐이 있습니다. 짐에는 {luggage_text} 등이 포함됩니다."
            
        except json.JSONDecodeError:
            return json_string

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
        
        # JSON 데이터를 자연어 텍스트로 변환
        scenario_text = self._convert_json_to_text(scenario_content.get('text'))

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
            
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        numbers = len(re.findall(r'\d', text))
        punctuation = len(re.findall(r'[^\w\s가-힣]', text))
        spaces = len(re.findall(r'\s', text))
        
        estimated = (
            korean_chars * 1.0 +
            english_chars * 0.25 +
            numbers * 0.5 +
            punctuation * 0.5 +
            spaces * 0.1
        )
        
        return max(1, int(estimated))
    
    def select_layout(self, custom_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """차량 좌석 레이아웃 선택 (오류 발생 시 예외를 발생시킴)"""
        try:
            options = custom_options if custom_options else self.load_option_list()
            scenario_content = self.load_scenario_content()
            
            prompt = self.create_optimized_prompt(options, scenario_content)
            
            content_parts = [prompt]
            if self.use_images and scenario_content.get('image'):
                content_parts.append(scenario_content['image'])
            
            print("Gemini API 처리 중...")
            start_time = time.time()
            
            safety_configurations = [
                [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ],
                [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
                ],
                [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                ]
            ]
            
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=800,
                temperature=0.2,
                top_p=0.8,
                top_k=40
            )
            
            response = None
            last_error = None
            
            for i, safety_settings in enumerate(safety_configurations):
                try:
                    if i > 0:
                        print(f"안전 설정 레벨 {i+1}로 재시도...")
                    
                    response = self.model.generate_content(
                        content_parts,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    if (response and 
                        hasattr(response, 'candidates') and response.candidates and 
                        len(response.candidates) > 0 and 
                        hasattr(response.candidates[0], 'content') and response.candidates[0].content and
                        hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts and
                        len(response.candidates[0].content.parts) > 0):
                        
                        finish_reason = response.candidates[0].finish_reason
                        if finish_reason == 1:
                            break
                        elif finish_reason == 2:
                            if i < len(safety_configurations) - 1:
                                continue
                            else:
                                raise Exception("모든 안전 설정에서 차단됨")
                        else:
                            print(f"경고: finish_reason = {finish_reason}")
                            break
                        
                except Exception as e:
                    last_error = e
                    logger.warning(f"안전 설정 레벨 {i+1} 실패: {e}")
                    if i == len(safety_configurations) - 1:
                        raise Exception(f"모든 시도 실패. 마지막 오류: {last_error}")
            
            if not response or not hasattr(response, 'candidates') or not response.candidates:
                raise Exception(f"API 응답 없음. 마지막 오류: {last_error}")
            
            content_text = response.text
            
            if not content_text or len(content_text.strip()) == 0:
                raise Exception("빈 응답 수신")
            
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(content_text)
            
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                try:
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                except AttributeError:
                    pass
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"토큰: {input_tokens}→{output_tokens} | 시간: {response_time:.2f}s")
            print("Gemini 응답:")
            print("-" * 30)
            print(content_text)
            print("-" * 30)
            
            return self._parse_response(content_text, options)
            
        except Exception as e:
            print(f"레이아웃 선택 오류: {e}")
            raise
    
    def _parse_response(self, response_text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """강화된 응답 파싱 (오류 발생 시 예외를 발생시킴)"""
        if not response_text:
            raise ValueError("빈 응답 텍스트가 제공되었습니다.")
        
        try:
            code_block_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)
            json_str = code_block_match.group(1) if code_block_match else response_text
            
            parsed_data = json.loads(json_str)
            
            if not isinstance(parsed_data, dict):
                raise ValueError("JSON의 최상위 구조가 객체(dict)가 아닙니다.")

            cell_layout = parsed_data.get('cell_layout')
            option_no = parsed_data.get('option_no')
            reason = parsed_data.get('reason')

            if cell_layout is None:
                raise ValueError("응답 JSON에 'cell_layout' 필드가 누락되었습니다.")
            if option_no is None:
                raise ValueError("응답 JSON에 'option_no' 필드가 누락되었습니다.")
            if reason is None:
                raise ValueError("응답 JSON에 'reason' 필드가 누락되었습니다.")

            if not isinstance(cell_layout, dict):
                raise TypeError(f"필드 'cell_layout'의 형식이 올바르지 않습니다: {type(cell_layout)}")
            try:
                option_no = int(option_no)
            except (ValueError, TypeError):
                raise ValueError(f"필드 'option_no'의 값이 유효한 정수가 아닙니다: {option_no}")
            if not isinstance(reason, str):
                raise TypeError(f"필드 'reason'의 형식이 올바르지 않습니다: {type(reason)}")
            
            print("JSON 파싱 성공")
            return {
                'cell_layout': cell_layout,
                'option_no': option_no,
                'reason': reason,
                'raw_response': response_text
            }
        
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 디코딩 오류: {e}") from e
        except Exception as e:
            raise RuntimeError(f"응답 파싱 중 예상치 못한 오류 발생: {e}") from e

def main():
    """최적화된 메인 함수"""
    try:
        print("=" * 60)
        print("최적화된 차량 좌석 레이아웃 선택 시스템 v2.5")
        print("=" * 60)
        
        selector = OptimizedGeminiSelector(use_images=False)
        
        result = selector.select_layout()
        
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
        print("1. 'tetris_secrets.json' 파일의 API 키 설정 확인")
        print("2. 인터넷 연결 상태 확인")
        print("3. 'SW/Chain2/main_v5/scenarios/test.txt' 파일 존재 여부 및 읽기 권한 확인")
        print("4. Google AI API 할당량 확인")

if __name__ == "__main__":
    main()