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

# JSON 스키마를 위한 Pydantic 모델 정의 (프롬프트에 맞춤)
class CircleCoords(BaseModel):
    center_x_percent: float = Field(ge=0, le=100)
    center_y_percent: float = Field(ge=0, le=100)
    diameter_percent: float = Field(ge=5, le=20)  # 프롬프트: width 대비 5~20%

class LuggageSelection(BaseModel):
    luggage_description: str = Field(description="선택된 짐의 설명")
    coordinates: CircleCoords = Field(description="선택된 짐의 중심 좌표(%)와 지름(%)")

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
        self.output_base = self.base_path / "chain1_out" / "chat1" / "single" / "gemini_ver"
        
        # 시나리오 폴더 목록
        self.scenarios = ["items_combined", "items_separated"]
        
        # 단일 프롬프트 로드 (파일에서)
        self.single_prompt = self._load_prompt("gemini_ver.txt")

    def _load_prompt(self, filename: str) -> str:
        """프롬프트 파일 로드 (프롬프트 자체는 수정하지 않음)"""
        prompt_file = self.prompt_path / filename
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def encode_image(self, image_path: Path):
        """이미지를 base64로 인코딩 + 원본 크기 반환 (성능을 위해 여기서 크기도 함께 반환)"""
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        image = Image.open(image_path)
        print(f"Base64 인코딩할 이미지 크기: {image.size}")
        return base64.b64encode(image_data).decode('utf-8'), image.size  # (b64, (w,h))

    def call_llm_with_single_prompt(self, model_name: str, image_path: Path, use_structured_output: bool = False):
        """단일 프롬프트로 LLM 호출 (프롬프트 앞에 이미지 크기 자동 주입)"""
        base64_image, (img_w, img_h) = self.encode_image(image_path)

        # 프롬프트 맨 앞에 이미지 크기 변수 정의를 추가 (불필요한 기호 없이, 변수명은 그대로 사용)
        # 예: {image_width}=1920\n{image_height}=1080\n\n<원본 프롬프트>
        prompt_with_dims = f"{{image_width}}={img_w}\n{{image_height}}={img_h}\n\n{self.single_prompt}"
        
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
                "Your primary task is to identify and provide the exact location of a specific piece of luggage in an image." 
                "Follow the instructions exactly and output only valid JSON with precise coordinates."
            ))
            
            user_msg = HumanMessage(content=[
                {"type": "text", "text": prompt_with_dims},
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
                    "Your primary task is to identify and provide the exact location of a specific piece of luggage in an image." 
                    "Follow the instructions exactly and output only valid JSON with precise coordinates."
                ),
                generation_config={"response_mime_type": "application/json"}
            )
            
            try:
                # 이미지 객체 (Gemini에 전달)
                image = Image.open(image_path)
                print(f"이미지 포맷: {image.format}, 모드: {image.mode}")
                
                # Gemini API 호출 (프롬프트 앞에 이미지 크기 주입한 버전 사용)
                response = model.generate_content([
                    {           
                        "role": "user",
                        "parts": [
                            {"text": prompt_with_dims},
                            image
                        ]
                    }
                ])
                return response.text
            except Exception as e:
                print(f"Gemini 모델 {model_name} 호출 실패: {e}")
                return None

    def parse_coordinates_from_response(self, response) -> Dict[str, Any]:
        """원 중심/지름 파싱 (프롬프트의 출력 포맷과 정확히 일치)"""
        try:
            def postprocess(obj: Dict[str, Any]) -> Dict[str, Any] | None:
                if not isinstance(obj, dict): return None
                if "luggage_description" not in obj or "coordinates" not in obj: return None

                coords = obj["coordinates"]
                required = ["center_x_percent", "center_y_percent", "diameter_percent"]
                if not all(k in coords for k in required): return None

                cx = float(coords["center_x_percent"])
                cy = float(coords["center_y_percent"])
                d  = float(coords["diameter_percent"])

                # 범위 검증 (프롬프트 제약과 일치)
                if not (0.0 <= cx <= 100.0 and 0.0 <= cy <= 100.0 and 5.0 <= d <= 20.0):
                    return None

                return {
                    "luggage_description": obj["luggage_description"],
                    "coordinates": {
                        "center_x_percent": cx,
                        "center_y_percent": cy,
                        "diameter_percent": d
                    }
                }

            # Pydantic 모델인 경우 (OpenAI 구조화된 출력)
            if isinstance(response, LuggageSelection):
                raw = {
                    "luggage_description": response.luggage_description,
                    "coordinates": response.coordinates.model_dump()
                }
                return postprocess(raw)
            
            # 텍스트(JSON) 응답인 경우
            if isinstance(response, str):
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
        """원 그리기 (중심 좌표 + 지름[% of width] 기반).
        디버그 로그로 원본 이미지 크기와 현재 그릴 이미지 크기를 함께 출력한다.
        """
        try:
            # 이미지 열기
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            # 이미지 크기
            width, height = image.size

            # 디버깅: 원본 이미지 크기와 현재 PIL 이미지 크기 비교
            print(f"=== 이미지 정보 ===")
            print(f"파일 경로: {image_path}")
            print(f"원본 이미지 크기: {Image.open(image_path).size}, PIL이 그릴 이미지 크기: {image.size}")

            if 'coordinates' in coordinates:
                coords = coordinates['coordinates']
                center_x = coords['center_x_percent'] * width / 100.0
                center_y = coords['center_y_percent'] * height / 100.0
                # 프롬프트: diameter_percent는 이미지 "width" 기준
                diameter_pixels = coords['diameter_percent'] * width / 100.0
                radius = diameter_pixels / 2.0

                print(
                    f"원: 중심({coords['center_x_percent']:.1f}%, {coords['center_y_percent']:.1f}%), "
                    f"지름 {coords['diameter_percent']:.1f}% of width "
                    f"=> 픽셀 중심({center_x:.1f}, {center_y:.1f}), 반지름 {radius:.1f}px"
                )

                # 빨간색 원 (테두리만)
                draw.ellipse([
                    center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius
                ], outline='red', width=12)

                # 중심점 표시 (작은 점)
                center_dot_radius = 3
                draw.ellipse([
                    center_x - center_dot_radius, center_y - center_dot_radius,
                    center_x + center_dot_radius, center_y + center_dot_radius
                ], fill='red', outline='red')

            # 이미지 저장
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, 'JPEG', quality=95)
            
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
        
        # OpenAI 모델은 구조화된 출력 사용, Gemini는 JSON 텍스트
        use_structured = model_name.startswith("gpt")
        response = self.call_llm_with_single_prompt(
            model_name, image_path, use_structured
        )
        
        if not response:
            print("LLM 호출 실패")
            return
            
        print(f"LLM 응답: {response}")
        
        # 좌표 파싱
        parsed = self.parse_coordinates_from_response(response)
        
        if not parsed:
            print("좌표 추출 실패")
            # 실패한 경우에도 텍스트 응답 저장
            text_output_path = model_output_dir / f"{image_name}_text.txt"
            self.save_text_response(str(response), text_output_path)
            return
        
        # 좌표 상세 분석 로그
        print(f"\n=== 추출된 좌표 분석 ===")
        print(f"설명: {parsed.get('luggage_description', 'N/A')}")
        coords = parsed['coordinates']
        print(f"center_x_percent: {coords['center_x_percent']}% (0-100)")
        print(f"center_y_percent: {coords['center_y_percent']}% (0-100)")
        print(f"diameter_percent: {coords['diameter_percent']}% of width (5-20)")
        
        # 가장자리 경고 (선택)
        if coords['center_x_percent'] < 5 or coords['center_x_percent'] > 95:
            print(f"⚠️  중심 X가 이미지 가장자리 근처: {coords['center_x_percent']}%")
        if coords['center_y_percent'] < 5 or coords['center_y_percent'] > 95:
            print(f"⚠️  중심 Y가 이미지 가장자리 근처: {coords['center_y_percent']}%")
        
        # 텍스트 응답 저장
        combined_response = (
            f"Single Prompt Circle Response:\n{response}\n\n"
            f"Parsed Coordinates:\n{json.dumps(parsed, indent=2, ensure_ascii=False)}"
        )
        text_output_path = model_output_dir / f"{image_name}_text.txt"
        self.save_text_response(combined_response, text_output_path)
        
        # 이미지에 원 그리기
        image_output_path = model_output_dir / f"{image_name}_image.jpeg"
        self.draw_circle_on_image(image_path, parsed, image_output_path)

    def process_all_images(self):
        """모든 이미지 처리"""
        print("Chain1 단일 프롬프트 분석 시작...")
        
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
