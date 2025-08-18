import os
import json
import base64
from pathlib import Path
from PIL import Image, ImageDraw
from langchain_openai import ChatOpenAI
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# tetris_secrets JSON 파일에서 API 키 로드
secrets_path = Path(__file__).parent / 'tetris_secrets.json'
with open(secrets_path, 'r', encoding='utf-8') as f:
    secrets = json.load(f)
openai_api_key = secrets['openai']['OPENAI_API_KEY']
google_api_key = secrets['google']['GOOGLE_API_KEY']

# Google Gemini API 키 설정
genai.configure(api_key=google_api_key)

# Pydantic 모델 정의 (퍼센트 기반)
class CenterPoint(BaseModel):
    center_x_percent: float = Field(description="중앙점 x 좌표 (0-100%)")
    center_y_percent: float = Field(description="중앙점 y 좌표 (0-100%)")
    diameter_percent: float = Field(description="원의 지름 (0-100%)")

class LuggageAnalysis(BaseModel):
    coordinates: CenterPoint = Field(description="선택된 짐의 중앙점 좌표와 원 지름")

class LuggageVolumeAnalyzer:
    def __init__(self):
        # API 키 설정
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        
        # 모델 목록
        self.models = ["gpt-4o", "gpt-4.1", "gemini-2.5-flash", "gemini-2.5-pro"]
        
        # 경로 설정 (현재 스크립트와 같은 폴더)
        self.base_path = Path(__file__).parent
        self.input_path = self.base_path / "chain1" / "chain1_image" / "1.jpg"  # "1.jpg"는 이미지 파일명
        self.output_coord_base = self.base_path / "chain1_out" / "2.1" / "output_coordinate"
        self.output_image_base = self.base_path / "chain1_out" / "2.1" / "output_image"

    def encode_image(self, image_path: Path) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_with_model(self, model_name: str, image_path: Path) -> dict:
        """특정 모델로 이미지 분석"""
        # Percentage-based prompt emphasizing complete inclusion (English)
        prompt = """You are a luggage volume estimation analyst. Analyze the provided image and identify the single piece of luggage that allows for the MOST ACCURATE VOLUME CALCULATION, considering all relevant factors.

Step-by-step instructions:
1. EVALUATE each luggage item for volume calculation accuracy by considering:
   - Shape regularity and geometric predictability
   - Visible surface information for depth/height estimation
   - Size relative to image resolution for measurement precision
   - Angle and perspective that aids volume assessment
   - Any visual cues that help determine 3D dimensions
   - Material/texture that provides depth information

2. SELECT the ONE luggage item that, based on your analysis, allows for the most precise volume calculation (regardless of visibility or occlusion issues)

3. LOCATE the selected luggage using PERCENTAGE coordinates (0-100%):
   - Think of the image as 100% width × 100% height
   - CRITICAL: Find the exact center point of the luggage
   - Determine an appropriate circle diameter to clearly mark the center
   - center_x_percent: the CENTER x coordinate of the luggage (0-100)
   - center_y_percent: the CENTER y coordinate of the luggage (0-100)
   - diameter_percent: circle diameter as percentage of image width (5-20% for visibility)

4. VERIFICATION - Check these requirements:
   - The circle should be centered on the luggage
   - Circle should be visible but not too large to cause confusion
   - center_x_percent and center_y_percent should be between 0 and 100
   - diameter_percent should be between 5 and 20

CRITICAL INSTRUCTION: The circle must be positioned at the exact geometric center of the luggage. Choose an appropriate diameter that makes it clear which luggage is selected without being too large or too small.

Example: If luggage center is at 40% width and 60% height with 10% diameter:
{"coordinates": {"center_x_percent": 40.0, "center_y_percent": 60.0, "diameter_percent": 10.0}}

Output format:
{"coordinates": {"center_x_percent": number, "center_y_percent": number, "diameter_percent": number}}"""
        
        # OpenAI 모델인지 Gemini 모델인지 구분
        if model_name.startswith("gpt"):
            # OpenAI 모델 사용
            llm = ChatOpenAI(
                model=model_name,
                api_key=self.openai_api_key,
                temperature=0
            )
            
            # 이미지 인코딩
            base64_image = self.encode_image(image_path)
            
            # 메시지 구성
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            )
            
            # LLM 호출 및 파싱
            try:
                response = llm.invoke([message])
                response_text = response.content
            except Exception as e:
                print(f"모델 {model_name} 분석 실패: {e}")
                return None
        else:
            # Gemini 모델 사용
            model = genai.GenerativeModel(model_name)
            
            # 이미지 열기
            image = Image.open(image_path)
            
            # Gemini API 호출
            try:
                response = model.generate_content([prompt, image])
                response_text = response.text
            except Exception as e:
                print(f"모델 {model_name} 분석 실패: {e}")
                return None

        # 공통 파싱 로직
        try:
            # JSON 파싱 시도
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                print(f"JSON을 찾을 수 없음: {response_text}")
                return None
        except Exception as e:
            print(f"모델 {model_name} 분석 실패: {e}")
            print(f"응답: {response_text if 'response_text' in locals() else 'No response'}")
            return None

    def save_coordinates(self, model_name: str, coordinates: dict, image_name: str):
        """좌표 정보를 JSON 파일로 저장"""
        model_dir = self.output_coord_base / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = model_dir / f"{image_name}_coordinates.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coordinates, f, indent=2, ensure_ascii=False)
        
        print(f"좌표 저장 완료: {output_file}")

    def draw_bounding_box(self, image_path: Path, coordinates: dict, model_name: str, image_name: str):
        """중앙점에 빨간 원을 그려서 이미지 저장"""
        try:
            # 이미지 열기
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            # 이미지 크기 얻기
            width, height = image.size
            
            # 퍼센트를 픽셀로 변환
            coords = coordinates['coordinates']
            center_x = int(coords['center_x_percent'] * width / 100)
            center_y = int(coords['center_y_percent'] * height / 100)
            diameter_pixels = int(coords['diameter_percent'] * width / 100)
            radius = diameter_pixels // 2
            
            print(f"이미지 크기: {width}x{height}")
            print(f"퍼센트 좌표: 중앙점({coords['center_x_percent']:.1f}%, {coords['center_y_percent']:.1f}%), 지름({coords['diameter_percent']:.1f}%)")
            print(f"픽셀 좌표: 중앙점({center_x}, {center_y}), 지름({diameter_pixels}px)")
            
            # 빨간색 원 그리기 (채워진 원)
            draw.ellipse([center_x - radius, center_y - radius, 
                         center_x + radius, center_y + radius], 
                        fill='red', outline='darkred', width=2)
            
            # 출력 디렉터리 생성
            model_dir = self.output_image_base / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 이미지 저장
            output_file = model_dir / f"{image_name}_circle.jpg"
            image.save(output_file, 'JPEG', quality=95)
            
            print(f"빨간 원 이미지 저장 완료: {output_file}")
            
        except Exception as e:
            print(f"원 그리기 실패: {e}")

    def process_images(self):
        """이미지 처리 메인 함수"""
        # 특정 이미지 파일 "1" 처리
        image_path = self.input_path
        
        if not image_path.exists():
            print(f"이미지를 찾을 수 없습니다: {image_path}")
            return
        
        print(f"처리할 이미지 발견: {image_path}")
        
        # 파일명
        image_name = image_path.name
        print(f"\n이미지 처리 중: {image_name}")
        
        for model_name in self.models:
            print(f"모델 {model_name}로 분석 중...")
            
            # 이미지 분석
            result = self.analyze_with_model(model_name, image_path)
            
            if result:
                # 좌표 저장
                self.save_coordinates(model_name, result, image_name)
                
                # 빨간 원 그리기
                self.draw_bounding_box(image_path, result, model_name, image_name)
            else:
                print(f"모델 {model_name} 분석 실패")

# 실행 코드
if __name__ == "__main__":
    analyzer = LuggageVolumeAnalyzer()
    analyzer.process_images()