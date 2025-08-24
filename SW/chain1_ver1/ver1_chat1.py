# 2단계 프롬프트 
'''
import os
import json
import base64
from pathlib import Path
from PIL import Image, ImageDraw
from langchain_openai import ChatOpenAI
import google.generativeai as genai
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import re

# JSON 스키마를 위한 Pydantic 모델 정의
class CoordinateData(BaseModel):
    center_x_percent: float = Field(description="중앙점 x 좌표 (0-100%)", ge=0, le=100)
    center_y_percent: float = Field(description="중앙점 y 좌표 (0-100%)", ge=0, le=100) 
    diameter_percent: float = Field(description="원의 지름 (5-12%)", ge=5, le=12)

class LuggageSelection(BaseModel):
    description: str = Field(description="선택된 짐의 설명")
    coordinates: CoordinateData = Field(description="선택된 짐의 중앙점 좌표와 원 지름")

class Chain1Ver1Analyzer:
    def __init__(self):
        # tetris_secrets JSON 파일에서 API 키 로드 (SW 폴더에 위치)
        secrets_path = Path(__file__).parent.parent / 'tetris_secrets.json'
        with open(secrets_path, 'r', encoding='utf-8') as f:
            secrets = json.load(f)
        
        self.openai_api_key = secrets['openai']['OPENAI_API_KEY']
        self.google_api_key = secrets['google']['GOOGLE_API_KEY']
        
        # Google Gemini API 키 설정
        genai.configure(api_key=self.google_api_key)
        
        # 모델 목록
        self.models = ["gpt-4o", "gpt-4.1", "gemini-2.5-flash", "gemini-2.5-pro"]
        
        # 경로 설정
        self.base_path = Path(__file__).parent
        self.image_base_path = self.base_path / "chain1_image"
        self.prompt_path = self.base_path / "chain1_prompt"
        self.output_base = self.base_path / "chain1_out" / "chat1" / "gpt_ver"
        
        # 시나리오 폴더 목록
        self.scenarios = ["items_combined", "items_separated"]
        
        # 프롬프트 로드
        self.step1_prompt = self._load_prompt("step1_gpt_ver.txt")
        self.step2_prompt = self._load_prompt("step2_gpt_ver.txt")

    def _load_prompt(self, filename: str) -> str:
        """프롬프트 파일 로드"""
        prompt_file = self.prompt_path / filename
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def create_combined_prompt(self) -> str:
        """Step1과 Step2 프롬프트를 결합한 통합 프롬프트 생성"""
        return f"""Please follow these steps in sequence:

Step 1:
{self.step1_prompt}

Step 2:
{self.step2_prompt}

Complete both steps and provide the final JSON output for Step 2."""

    def encode_image(self, image_path: Path) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def create_message_for_model(self, image_path: Path, model_name: str) -> List:
        """모델별 최적화된 메시지 생성"""
        base64_image = self.encode_image(image_path)
        
        if model_name.startswith("gpt"):
            # OpenAI용 - 단순한 구조
            system_msg = SystemMessage(content=(
                "You are a precise luggage analysis expert. "
                "Follow the instructions exactly and output only valid JSON with precise coordinates."
            ))
            
            # 통합 프롬프트 생성
            combined_prompt = self.create_combined_prompt()
            
            user_msg = HumanMessage(content=[
                {"type": "text", "text": combined_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ])
            
            return [system_msg, user_msg]
        else:
            # Gemini용 - 텍스트와 이미지 분리
            combined_prompt = self.create_combined_prompt()
            return [combined_prompt, base64_image]

    def call_llm_with_optimized_prompt(self, model_name: str, image_path: Path, use_structured_output: bool = False):
        """최적화된 프롬프트로 LLM 호출"""
        if model_name.startswith("gpt"):
            # OpenAI 모델 사용
            if use_structured_output:
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=self.openai_api_key,
                    temperature=0,
                    max_tokens=1000
                ).with_structured_output(LuggageSelection)
            else:
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=self.openai_api_key,
                    temperature=0,
                    max_tokens=1000
                )
            
            messages = self.create_message_for_model(image_path, model_name)
            
            try:
                response = llm.invoke(messages)
                return response if use_structured_output else response.content
            except Exception as e:
                print(f"OpenAI 모델 {model_name} 호출 실패: {e}")
                return None
                
        else:
            # Gemini 모델 사용
            model = genai.GenerativeModel(model_name)
            
            try:
                # 이미지 열기
                image = Image.open(image_path)
                
                # Gemini API 호출
                combined_prompt = self.create_combined_prompt()
                response = model.generate_content([combined_prompt, image])
                return response.text
            except Exception as e:
                print(f"Gemini 모델 {model_name} 호출 실패: {e}")
                return None

    def parse_coordinates_from_response(self, response) -> Dict[str, Any]:
        """고정밀 좌표 파싱"""
        try:
            # Pydantic 모델인 경우 (OpenAI 구조화된 출력)
            if isinstance(response, LuggageSelection):
                return {
                    "description": response.description,
                    "coordinates": {
                        "center_x_percent": float(response.coordinates.center_x_percent),
                        "center_y_percent": float(response.coordinates.center_y_percent),
                        "diameter_percent": float(response.coordinates.diameter_percent)
                    }
                }
            
            # 텍스트 응답인 경우
            if isinstance(response, str):
                # JSON 블록 찾기 (더 정확한 패턴)
                json_patterns = [
                    r'\{[^{}]*"coordinates"[^{}]*\{[^{}]*\}[^{}]*\}',  # 단일 레벨 JSON
                    r'\{.*?"coordinates".*?\{.*?\}.*?\}',  # 중첩 JSON
                    r'\{.*\}',  # 일반 JSON
                ]
                
                result = None
                for pattern in json_patterns:
                    json_match = re.search(pattern, response, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group())
                            break
                        except:
                            continue
                
                if not result:
                    print(f"JSON을 찾을 수 없음: {response}")
                    return None
                
                # 구조 및 범위 검증
                if 'coordinates' in result and isinstance(result['coordinates'], dict):
                    coords = result['coordinates']
                    
                    # 필수 키 확인
                    required_keys = ['center_x_percent', 'center_y_percent', 'diameter_percent']
                    if not all(key in coords for key in required_keys):
                        print(f"필수 좌표 키 누락: {coords}")
                        return None
                    
                    # 범위 검증 (작은 원 범위)
                    if (0 <= float(coords['center_x_percent']) <= 100 and 
                        0 <= float(coords['center_y_percent']) <= 100 and 
                        5 <= float(coords['diameter_percent']) <= 12):
                        
                        # float 정밀도 보장
                        result['coordinates'] = {
                            'center_x_percent': float(coords['center_x_percent']),
                            'center_y_percent': float(coords['center_y_percent']),
                            'diameter_percent': float(coords['diameter_percent'])
                        }
                        return result
                    else:
                        print(f"좌표 값이 범위를 벗어남: {coords}")
                        return None
                else:
                    print(f"JSON 구조가 올바르지 않음: {result}")
                    return None
            
            print(f"알 수 없는 응답 형태: {type(response)}")
            return None
            
        except Exception as e:
            print(f"좌표 파싱 실패: {e}")
            print(f"응답: {response}")
            return None

    def draw_circle_on_image(self, image_path: Path, coordinates: Dict[str, Any], output_path: Path):
        """고정밀 원 그리기 + 디버깅 정보"""
        try:
            # 이미지 열기
            image = Image.open(image_path)
            
            # 이미지 정보 상세 출력
            print(f"=== 이미지 정보 ===")
            print(f"파일 경로: {image_path}")
            print(f"이미지 모드: {image.mode}")
            print(f"이미지 크기: {image.size}")
            print(f"이미지 포맷: {image.format}")
            
            draw = ImageDraw.Draw(image)
            
            # 이미지 크기
            width, height = image.size
            
            # 고정밀 퍼센트-픽셀 변환 (소수점 유지)
            coords = coordinates['coordinates']
            center_x_percent = coords['center_x_percent']
            center_y_percent = coords['center_y_percent']
            diameter_percent = coords['diameter_percent']
            
            center_x = center_x_percent * width / 100.0
            center_y = center_y_percent * height / 100.0
            diameter_pixels = diameter_percent * width / 100.0
            radius = diameter_pixels / 2.0
            
            print(f"\n=== 좌표 변환 정보 ===")
            print(f"원본 퍼센트 좌표: 중앙점({center_x_percent:.3f}%, {center_y_percent:.3f}%), 지름({diameter_percent:.3f}%)")
            print(f"변환된 픽셀 좌표: 중앙점({center_x:.3f}, {center_y:.3f}), 지름({diameter_pixels:.3f}px)")
            print(f"원 경계 좌표: 좌상({center_x - radius:.1f}, {center_y - radius:.1f}) ~ 우하({center_x + radius:.1f}, {center_y + radius:.1f})")
            
            # 이미지 경계 검증
            if center_x < 0 or center_x > width or center_y < 0 or center_y > height:
                print(f"⚠️  경고: 중심점이 이미지 범위를 벗어남!")
            
            if center_x - radius < 0 or center_x + radius > width or center_y - radius < 0 or center_y + radius > height:
                print(f"⚠️  경고: 원이 이미지 경계를 벗어남!")
            
            # 작은 원 그리기 (중심점 표시용, 두꺼운 테두리)
            draw.ellipse([
                center_x - radius, center_y - radius, 
                center_x + radius, center_y + radius
            ], fill=None, outline='red', width=12)  # 두꺼운 테두리로 가시성 확보
            
            # 중심점도 표시 (작은 점)
            center_dot_radius = 3
            draw.ellipse([
                center_x - center_dot_radius, center_y - center_dot_radius,
                center_x + center_dot_radius, center_y + center_dot_radius
            ], fill='red', outline='red')
            
            # 십자가 표시 (정확한 중심점 확인용)
            cross_size = 15
            draw.line([center_x - cross_size, center_y, center_x + cross_size, center_y], fill='blue', width=3)
            draw.line([center_x, center_y - cross_size, center_x, center_y + cross_size], fill='blue', width=3)
            
            # 이미지 저장 전 크기 재확인
            print(f"\n=== 저장 전 이미지 정보 ===")
            print(f"저장 전 이미지 크기: {image.size}")
            
            # 이미지 저장
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, 'JPEG', quality=95)
            
            # 저장된 이미지 크기 확인
            saved_image = Image.open(output_path)
            print(f"저장된 이미지 크기: {saved_image.size}")
            
            if image.size != saved_image.size:
                print(f"⚠️  경고: 저장 과정에서 이미지 크기가 변경됨!")
            
            print(f"원 그리기 완료: {output_path}")
            
        except Exception as e:
            print(f"원 그리기 실패: {e}")
            import traceback
            traceback.print_exc()

    def save_text_response(self, text: str, output_path: Path):
        """텍스트 응답 저장"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"텍스트 저장 완료: {output_path}")

    def process_single_image(self, model_name: str, scenario: str, image_name: str):
        """단일 이미지 처리"""
        print(f"\n{'='*60}")
        print(f"처리 중: {model_name} | {scenario} | {image_name}")
        print(f"{'='*60}")
        
        # 이미지 경로 설정
        image_path = self.image_base_path / scenario / f"{image_name}.jpg"
        
        if not image_path.exists():
            print(f"이미지를 찾을 수 없습니다: {image_path}")
            return
        
        # 출력 경로 설정
        model_output_dir = self.output_base / model_name / scenario
        
        # LLM 호출 (통합 프롬프트로 한 번에 처리)
        print(f"LLM 호출 중 (통합 프롬프트)...")
        
        # OpenAI 모델은 구조화된 출력 사용, Gemini는 일반 텍스트
        use_structured = model_name.startswith("gpt")
        response = self.call_llm_with_optimized_prompt(
            model_name, image_path, use_structured
        )
        
        if not response:
            print("LLM 호출 실패")
            return
            
        print(f"LLM 응답: {response}")
        
        # 좌표 파싱
        coordinates = self.parse_coordinates_from_response(response)
        
        if not coordinates:
            print("좌표 추출 실패")
            # 실패한 경우에도 텍스트 응답 저장
            text_output_path = model_output_dir / f"{image_name}_text.txt"
            self.save_text_response(str(response), text_output_path)
            return
        
        # 좌표 상세 분석
        print(f"\n=== 추출된 좌표 분석 ===")
        print(f"설명: {coordinates.get('description', 'N/A')}")
        coords = coordinates['coordinates']
        print(f"X 좌표: {coords['center_x_percent']}% (0-100% 범위)")
        print(f"Y 좌표: {coords['center_y_percent']}% (0-100% 범위)")
        print(f"원 지름: {coords['diameter_percent']}% (5-12% 범위)")
        
        # 좌표 합리성 검증
        if coords['center_x_percent'] < 10 or coords['center_x_percent'] > 90:
            print(f"⚠️  X 좌표가 이미지 가장자리에 위치: {coords['center_x_percent']}%")
        if coords['center_y_percent'] < 10 or coords['center_y_percent'] > 90:
            print(f"⚠️  Y 좌표가 이미지 가장자리에 위치: {coords['center_y_percent']}%")
        
        # 텍스트 응답 저장
        combined_response = f"LLM Response:\n{response}\n\nParsed Coordinates:\n{json.dumps(coordinates, indent=2, ensure_ascii=False)}"
        text_output_path = model_output_dir / f"{image_name}_text.txt"
        self.save_text_response(combined_response, text_output_path)
        
        # 이미지에 원 그리기
        image_output_path = model_output_dir / f"{image_name}_image.jpg"
        self.draw_circle_on_image(image_path, coordinates, image_output_path)

    def process_all_images(self):
        """모든 이미지 처리"""
        print("Chain1_Ver1 분석 시작 (통합 프롬프트 방식)...")
        
        # 각 모델별로 처리
        for model_name in self.models:
            print(f"\n{'='*80}")
            print(f"모델: {model_name} 처리 시작")
            print(f"{'='*80}")
            
            # 각 시나리오별로 처리
            for scenario in self.scenarios:
                print(f"\n시나리오: {scenario}")
                
                # 이미지 파일 "1", "2" 처리
                for image_name in ["1", "2"]:
                    try:
                        self.process_single_image(model_name, scenario, image_name)
                    except Exception as e:
                        print(f"이미지 처리 실패 ({model_name}/{scenario}/{image_name}): {e}")
                        continue
        
        print("\n" + "="*80)
        print("모든 처리 완료!")
        print("="*80)

# 실행 코드
if __name__ == "__main__":
    analyzer = Chain1Ver1Analyzer()
    analyzer.process_all_images()
'''

# 단일 프롬프트 (코드 수정 필요)
import os
import json
import base64
from pathlib import Path
from PIL import Image, ImageDraw
from langchain_openai import ChatOpenAI
import google.generativeai as genai
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import re

# JSON 스키마를 위한 Pydantic 모델 정의 (원 좌표)
class CoordinateData(BaseModel):
    anchor_x_percent: float = Field(ge=0, le=100)
    anchor_y_percent: float = Field(ge=0, le=100)
    max_safe_diameter_percent: float = Field(ge=2, le=10)

class LuggageSelection(BaseModel):
    description: str = Field(description="선택된 짐의 설명")
    coordinates: CoordinateData = Field(description="선택된 짐의 앵커 좌표와 원 지름")

class Chain1SinglePromptAnalyzer:
    def __init__(self):
        # tetris_secrets JSON 파일에서 API 키 로드 (SW 폴더에 위치)
        secrets_path = Path(__file__).parent.parent / 'tetris_secrets.json'
        with open(secrets_path, 'r', encoding='utf-8') as f:
            secrets = json.load(f)
        
        self.openai_api_key = secrets['openai']['OPENAI_API_KEY']
        self.google_api_key = secrets['google']['GOOGLE_API_KEY']
        
        # Google Gemini API 키 설정
        genai.configure(api_key=self.google_api_key)
        
        # 모델 목록
        self.models = ["gpt-4o", "gpt-4.1", "gemini-2.5-flash", "gemini-2.5-pro"]
        
        # 경로 설정
        self.base_path = Path(__file__).parent
        self.image_base_path = self.base_path / "chain1_image"
        self.prompt_path = self.base_path / "chain1_prompt"
        self.output_base = self.base_path / "chain1_out" / "chat1" / "single" / "gpt_ver"
        
        # 시나리오 폴더 목록
        self.scenarios = ["items_combined", "items_separated"]
        
        # 단일 프롬프트 로드 (파일에서)
        self.single_prompt = self._load_prompt("gpt_ver.txt")

    def _load_prompt(self, filename: str) -> str:
        """프롬프트 파일 로드"""
        prompt_file = self.prompt_path / filename
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def encode_image(self, image_path: Path) -> str:
        """이미지를 base64로 인코딩 + 크기 정보 로깅"""
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # 원본 이미지 크기 확인
        image = Image.open(image_path)
        print(f"Base64 인코딩할 이미지 크기: {image.size}")
        
        return base64.b64encode(image_data).decode('utf-8')

    def call_llm_with_single_prompt(self, model_name: str, image_path: Path, use_structured_output: bool = False):
        """단일 프롬프트로 LLM 호출"""
        base64_image = self.encode_image(image_path)
        
        if model_name.startswith("gpt"):
            # OpenAI 모델 사용
            if use_structured_output:
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=self.openai_api_key,
                    temperature=0,
                    max_tokens=1000
                ).with_structured_output(LuggageSelection)
            else:
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=self.openai_api_key,
                    temperature=0,
                    max_tokens=1000
                )
            
            # 메시지 구성
            system_msg = SystemMessage(content=(
                "You are a precise luggage analysis expert. "
                "Follow the instructions exactly and output only valid JSON with precise coordinates."
            ))
            
            user_msg = HumanMessage(content=[
                {"type": "text", "text": self.single_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ])
            
            try:
                response = llm.invoke([system_msg, user_msg])
                return response if use_structured_output else response.content
            except Exception as e:
                print(f"OpenAI 모델 {model_name} 호출 실패: {e}")
                return None
                
        else:
            # Gemini 모델 사용
            model = genai.GenerativeModel(
                model_name,
                system_instruction=(
                    "You are a precise luggage analysis expert. "
                    "Follow the instructions exactly and output only valid JSON with precise coordinates."
                ), generation_config={"response_mime_type": "application/json"}
            )
            
            try:
                # 이미지 열기 (이제 JPEG이므로 변환 불필요)
                image = Image.open(image_path)
                print(f"이미지 포맷: {image.format}, 모드: {image.mode}")
                
                # Gemini API 호출
                response = model.generate_content([
                    {           
                        "role": "user",
                        "parts": [
                            {"text": self.single_prompt},image]  # PIL.Image 객체 그대로 전달 가능
                    }
            ])
                return response.text
            except Exception as e:
                print(f"Gemini 모델 {model_name} 호출 실패: {e}")
                return None

    def parse_coordinates_from_response(self, response) -> Dict[str, Any]:
        """앵커 좌표 파싱"""
        try:
            def postprocess(obj: Dict[str, Any]) -> Dict[str, Any] | None:
                if not isinstance(obj, dict): return None
                if "description" not in obj or "coordinates" not in obj:
                    return None

                coords = obj["coordinates"]
                required_coords = ["anchor_x_percent","anchor_y_percent","max_safe_diameter_percent"]
                if not all(k in coords for k in required_coords): return None

                # float 변환
                x = float(coords["anchor_x_percent"])
                y = float(coords["anchor_y_percent"])
                d = float(coords["max_safe_diameter_percent"])

                # 관계 검증
                if not (0 <= x <= 100 and 0 <= y <= 100 and 2 <= d <= 10): return None

                return {
                    "description": obj["description"],
                    "coordinates": {"anchor_x_percent": x, "anchor_y_percent": y, "max_safe_diameter_percent": d}
                }

            # Pydantic 모델인 경우 (OpenAI 구조화된 출력)
            if isinstance(response, LuggageSelection):
                raw = {
                    "description": response.description,
                    "coordinates": response.coordinates.model_dump()
                }
                return postprocess(raw)
            
            # 텍스트 응답인 경우
            if isinstance(response, str):
                # JSON 블록 찾기
                cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response).strip()
                candidates = re.findall(r"\{.*?\}", cleaned, re.DOTALL)
                for c in candidates + [cleaned]:
                    try:
                        obj = json.loads(c)
                    except Exception:
                        continue
                    out = postprocess(obj)
                    if out is not None:
                        return out
                print(f"유효 JSON을 찾지 못함: {response[:200]}...")
                return None
            
            print(f"알 수 없는 응답 형태: {type(response)}")
            return None
            
        except Exception as e:
            print(f"좌표 파싱 실패: {e}")
            print(f"응답: {response}")
            return None

    def draw_circle_on_image(self, image_path: Path, coordinates: Dict[str, Any], output_path: Path):
        """앵커 원 그리기"""
        try:
            # 이미지 열기
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            # 이미지 크기
            width, height = image.size
            
            print(f"=== 이미지 정보 ===")
            print(f"파일 경로: {image_path}")
            print(f"이미지 크기: {image.size}")
            
            # 앵커 원 그리기
            coords = coordinates['coordinates']
            center_x = coords['anchor_x_percent'] * width / 100.0
            center_y = coords['anchor_y_percent'] * height / 100.0
            diameter_pixels = coords['max_safe_diameter_percent'] * min(width, height) / 100.0
            radius = diameter_pixels / 2.0
            
            print(f"앵커: ({coords['anchor_x_percent']:.1f}%, {coords['anchor_y_percent']:.1f}%), 지름: {coords['max_safe_diameter_percent']:.1f}%")
            print(f"픽셀 좌표: 중심({center_x:.1f}, {center_y:.1f}), 반지름: {radius:.1f}px")
            
            # 빨간색 원 그리기 (테두리만)
            draw.ellipse([
                center_x - radius, center_y - radius, 
                center_x + radius, center_y + radius
            ], fill=None, outline='red', width=12)
            
            # 중심점 표시 (작은 점)
            center_dot_radius = 3
            draw.ellipse([
                center_x - center_dot_radius, center_y - center_dot_radius,
                center_x + center_dot_radius, center_y + center_dot_radius
            ], fill='red', outline='red')
            
            # 이미지 저장
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, 'JPEG', quality=95)
            
            print(f"앵커 원 그리기 완료: {output_path}")
            
        except Exception as e:
            print(f"원 그리기 실패: {e}")
            import traceback
            traceback.print_exc()

    def save_text_response(self, text: str, output_path: Path):
        """텍스트 응답 저장"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"텍스트 저장 완료: {output_path}")

    def process_single_image(self, model_name: str, scenario: str, image_name: str):
        """단일 이미지 처리 (단일 프롬프트 방식)"""
        print(f"\n{'='*60}")
        print(f"처리 중: {model_name} | {scenario} | {image_name}")
        print(f"{'='*60}")
        
        # 이미지 경로 설정 (.jpeg 확장자 사용)
        image_path = self.image_base_path / scenario / f"{image_name}.jpeg"
        
        # 디버깅: 경로 및 파일 존재 확인
        print(f"찾는 이미지 경로: {image_path}")
        print(f"base_path: {self.base_path}")
        print(f"image_base_path: {self.image_base_path}")
        print(f"scenario 폴더 존재: {(self.image_base_path / scenario).exists()}")
        
        # scenario 폴더에 있는 파일들 나열
        scenario_folder = self.image_base_path / scenario
        if scenario_folder.exists():
            files = list(scenario_folder.glob("*"))
            print(f"scenario 폴더 내 파일들: {files}")
        else:
            print(f"scenario 폴더가 존재하지 않음: {scenario_folder}")
        
        if not image_path.exists():
            print(f"이미지를 찾을 수 없습니다: {image_path}")
            return
        
        # 출력 경로 설정
        model_output_dir = self.output_base / model_name / scenario
        
        # LLM 호출 (단일 프롬프트로 처리)
        print(f"LLM 호출 중 (단일 프롬프트)...")
        
        # OpenAI 모델은 구조화된 출력 사용, Gemini는 일반 텍스트
        use_structured = model_name.startswith("gpt")
        response = self.call_llm_with_single_prompt(
            model_name, image_path, use_structured
        )
        
        if not response:
            print("LLM 호출 실패")
            return
            
        print(f"LLM 응답: {response}")
        
        # 좌표 파싱
        coordinates = self.parse_coordinates_from_response(response)
        
        if not coordinates:
            print("좌표 추출 실패")
            # 실패한 경우에도 텍스트 응답 저장
            text_output_path = model_output_dir / f"{image_name}_text.txt"
            self.save_text_response(str(response), text_output_path)
            return
        
        # 좌표 상세 분석
        print(f"\n=== 추출된 좌표 분석 ===")
        print(f"설명: {coordinates.get('description', 'N/A')}")
        coords = coordinates['coordinates']
        print(f"앵커 X: {coords['anchor_x_percent']}% (0-100% 범위)")
        print(f"앵커 Y: {coords['anchor_y_percent']}% (0-100% 범위)")
        print(f"안전 지름: {coords['max_safe_diameter_percent']}% (2-10% 범위)")
        
        # 좌표 합리성 검증
        if coords['anchor_x_percent'] < 10 or coords['anchor_x_percent'] > 90:
            print(f"⚠️  X 좌표가 이미지 가장자리에 위치: {coords['anchor_x_percent']}%")
        if coords['anchor_y_percent'] < 10 or coords['anchor_y_percent'] > 90:
            print(f"⚠️  Y 좌표가 이미지 가장자리에 위치: {coords['anchor_y_percent']}%")
        
        # 텍스트 응답 저장
        combined_response = f"Single Prompt Anchor Response:\n{response}\n\nParsed Coordinates:\n{json.dumps(coordinates, indent=2, ensure_ascii=False)}"
        text_output_path = model_output_dir / f"{image_name}_text.txt"
        self.save_text_response(combined_response, text_output_path)
        
        # 이미지에 앵커 원 그리기
        image_output_path = model_output_dir / f"{image_name}_image.jpeg"
        self.draw_circle_on_image(image_path, coordinates, image_output_path)

    def process_all_images(self):
        """모든 이미지 처리"""
        print("Chain1 단일 프롬프트 BBox 분석 시작...")
        
        # 각 모델별로 처리
        for model_name in self.models:
            print(f"\n{'='*80}")
            print(f"모델: {model_name} 처리 시작")
            print(f"{'='*80}")
            
            # 각 시나리오별로 처리
            for scenario in self.scenarios:
                print(f"\n시나리오: {scenario}")
                
                # 이미지 파일 "1", "2" 처리
                for image_name in ["1", "2"]:
                    try:
                        self.process_single_image(model_name, scenario, image_name)
                    except Exception as e:
                        print(f"이미지 처리 실패 ({model_name}/{scenario}/{image_name}): {e}")
                        continue
        
        print("\n" + "="*80)
        print("모든 처리 완료!")
        print("="*80)

# 실행 코드
if __name__ == "__main__":
    analyzer = Chain1SinglePromptAnalyzer()
    analyzer.process_all_images()