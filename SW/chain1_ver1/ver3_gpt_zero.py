import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

# JSON 스키마 정의 (정확한 구조)
class LuggageDetail(BaseModel):
    object: str
    object_confidence: float
    special_note: Optional[str] = None
    special_note_confidence: Optional[float] = None

class TotalLuggage(BaseModel):
    count: int
    confidence: float

class Largest(BaseModel):
    object: str
    largest_confidence: float

class Smallest(BaseModel):
    object: str
    smallest_confidence: float

class Extremes(BaseModel):
    largest: Largest
    smallest: Smallest

class LuggageAnalysisResult(BaseModel):
    total_luggage: TotalLuggage
    luggage_details: Dict[str, LuggageDetail]
    extremes: Extremes

class LuggageAnalyzer:
    def __init__(self):
        # API 키 로드 (모델 초기화 전에 먼저 실행)
        self._load_api_keys()
        
        # 정확한 경로 설정 (Desktop 경로 사용)
        desktop_path = Path.home() / "Desktop"
        self.base_path = desktop_path / "AIRL_ATM" / "SW" / "chain1_ver1"
        self.prompt1_path = self.base_path / "chain1_prompt" / "ver3_gpt_zero_1.txt"
        self.prompt2_path = self.base_path / "chain1_prompt" / "ver3_gpt_zero_2.txt"
        self.image_base_path = self.base_path / "chain1_image"
        self.output_base_path = self.base_path / "chain1_out" / "ver3" / "gpt_zero"
        
        # 프롬프트 로드
        self.stage1_prompt = self._load_prompt(self.prompt1_path)
        self.stage2_prompt = self._load_prompt(self.prompt2_path)
        
        # JSON 파서 설정
        self.parser = JsonOutputParser(pydantic_object=LuggageAnalysisResult)
        
        # API 키 로드 후 모델 초기화
        print("🤖 모델 초기화 중...")
        self.models = {
            "gemini-2.5-flash": ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
            "gemini-2.5-pro": ChatGoogleGenerativeAI(model="gemini-2.5-pro"),
            "gpt-4o": ChatOpenAI(model="gpt-4o"),
            "gpt-4.1": ChatOpenAI(model="gpt-4.1")
        }
        print("✅ 모든 모델 초기화 완료")
    
    def _load_api_keys(self):
        """tetris_secrets.json 파일에서 API 키 로드"""
        # 여러 위치에서 tetris_secrets.json 파일 탐색
        possible_paths = [
            Path.home() / "Desktop" / "AIRL_ATM" / "SW" / "tetris_secrets.json",
            Path.home() / "Desktop" / "AIRL_ATM" / "tetris_secrets.json",
            Path("C:/Users/User/Desktop/AIRL_ATM/SW/tetris_secrets.json"),
            Path("C:/Users/User/Desktop/AIRL_ATM/tetris_secrets.json")
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
            
            # 환경 변수 설정
            os.environ["OPENAI_API_KEY"] = secrets["openai"]["OPENAI_API_KEY"]
            os.environ["GOOGLE_API_KEY"] = secrets["google"]["GOOGLE_API_KEY"]
            
            print("🔑 API 키 로드 완료")
            
        except KeyError as e:
            raise KeyError(f"🚨 API 키를 찾을 수 없습니다: {e}")
        except json.JSONDecodeError:
            raise ValueError("🚨 tetris_secrets.json 파일 형식이 올바르지 않습니다")
        
    def _load_prompt(self, prompt_path: Path) -> str:
        """프롬프트 파일 로드"""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"🔧 프롬프트 로드 완료: {prompt_path.name}")
            return content
            
        except FileNotFoundError:
            raise FileNotFoundError(f"🚨 프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
    
    def _encode_image(self, image_path: Path) -> str:
        """이미지를 base64로 인코딩 (원본 크기/품질 유지)"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _get_image_paths(self) -> List[Path]:
        """이미지 경로 수집"""
        image_paths = []
        
        # items_combined 폴더에서 1.jpeg, 2.jpeg 찾기
        combined_folder = self.image_base_path / "items_combined"
        if combined_folder.exists():
            for i in range(1, 3):
                image_path = combined_folder / f"{i}.jpeg"
                if image_path.exists():
                    image_paths.append(image_path)
        
        # items_separated 폴더에서 1.jpeg, 2.jpeg 찾기
        separated_folder = self.image_base_path / "items_separated"
        if separated_folder.exists():
            for i in range(1, 3):
                image_path = separated_folder / f"{i}.jpeg"
                if image_path.exists():
                    image_paths.append(image_path)
        
        return image_paths
    
    def _create_stage1_chain(self, model_name: str):
        """1단계 추론 체인 생성"""
        model = self.models[model_name]
        
        prompt = ChatPromptTemplate.from_messages([
            ("human", [
                {"type": "text", "text": self.stage1_prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}
            ])
        ])
        
        chain = prompt | model
        return chain
    
    def _create_stage2_chain(self, model_name: str):
        """2단계 JSON 출력 체인 생성 (강력한 JSON 강제)"""
        model = self.models[model_name]
        
        # JSON 강제를 위한 구조화된 출력 설정
        if "gpt" in model_name:
            # OpenAI 모델의 경우 response_format으로 JSON 강제
            model = model.bind(response_format={"type": "json_object"})
        
        # JSON 스키마 문자열의 중괄호를 이스케이프 처리
        schema_str = self.parser.get_format_instructions()
        escaped_schema = schema_str.replace("{", "{{").replace("}", "}}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You must respond with ONLY valid JSON matching this exact schema: {escaped_schema}"),
            ("human", "{{stage1_output}}\n\n{{stage2_prompt}}\n\n**CRITICAL: Output ONLY valid JSON. No markdown, explanations, or code fences.**")
        ])
        
        chain = prompt | model | self.parser
        return chain
    
    def analyze_image(self, image_path: Path, model_name: str) -> Dict[str, Any]:
        """2단계 이미지 분석 (강화된 오류 처리)"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 1단계: 추론 추출
                stage1_chain = self._create_stage1_chain(model_name)
                image_data = self._encode_image(image_path)
                
                stage1_result = stage1_chain.invoke({"image_data": image_data})
                stage1_output = stage1_result.content if hasattr(stage1_result, 'content') else str(stage1_result)
                
                print(f"1단계 추론 완료 (길이: {len(stage1_output)})")
                
                # 2단계: JSON 구조화된 출력
                stage2_chain = self._create_stage2_chain(model_name)
                
                result = stage2_chain.invoke({
                    "stage1_output": stage1_output,
                    "stage2_prompt": self.stage2_prompt
                })
                
                # JSON 구조 검증
                if self._validate_json_structure(result):
                    return result
                else:
                    print(f"⚠️ JSON 구조 검증 실패 (시도 {attempt + 1}/{max_retries})")
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON 파싱 오류 (시도 {attempt + 1}/{max_retries}): {e}")
            except Exception as e:
                print(f"⚠️ 분석 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                
            if attempt < max_retries - 1:
                print(f"🔄 재시도 중...")
        
        print(f"❌ 최대 재시도 횟수 초과")
        return None
    
    def _validate_json_structure(self, result: Dict[str, Any]) -> bool:
        """JSON 구조 검증"""
        try:
            # 필수 키 확인
            required_keys = ["total_luggage", "luggage_details", "extremes"]
            for key in required_keys:
                if key not in result:
                    print(f"❌ 필수 키 누락: {key}")
                    return False
            
            # total_luggage 구조 확인
            tl = result["total_luggage"]
            if not all(k in tl for k in ["count", "confidence"]):
                print("❌ total_luggage 구조 오류")
                return False
                
            # extremes 구조 확인
            ex = result["extremes"]
            if not all(k in ex for k in ["largest", "smallest"]):
                print("❌ extremes 구조 오류")
                return False
                
            print("✅ JSON 구조 검증 통과")
            return True
            
        except Exception as e:
            print(f"❌ JSON 구조 검증 중 오류: {e}")
            return False
    
    def save_result(self, result: Dict[str, Any], model_name: str, image_path: Path):
        """결과 저장"""
        # 모델명 폴더 생성
        model_output_dir = self.output_base_path / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 생성 (폴더명_이미지명.json)
        folder_name = image_path.parent.name  # items_combined 또는 items_separated
        image_name = image_path.stem  # 1 또는 2
        filename = f"{folder_name}_{image_name}.json"
        
        # 저장
        output_path = model_output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 저장 완료: {output_path}")
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("🚀 === 2단계 LangChain 짐 분석 시작 ===")
        
        # 이미지 경로 수집
        image_paths = self._get_image_paths()
        
        if not image_paths:
            print("❌ 분석할 이미지를 찾을 수 없습니다.")
            print(f"📁 확인 경로: {self.image_base_path}")
            return
        
        print(f"📸 발견된 이미지: {len(image_paths)}개")
        for img_path in image_paths:
            print(f"  - {img_path}")
        
        # 지정된 4개 모델로 각 이미지 분석
        for model_name in ["gemini-2.5-flash", "gemini-2.5-pro", "gpt-4o", "gpt-4.1"]:
            print(f"\n🤖 --- {model_name} 모델 분석 중 ---")
            
            for image_path in image_paths:
                print(f"⚡ 분석 중: {image_path.name} ({image_path.parent.name})")
                
                # 분석 실행
                result = self.analyze_image(image_path, model_name)
                
                if result:
                    # 결과 저장
                    self.save_result(result, model_name, image_path)
                    print("✅ 분석 및 저장 완료")
                else:
                    print("❌ 분석 실패")
        
        print("\n🎉 === 전체 분석 완료 ===")

# 실행 함수
def main():
    """메인 실행 함수"""
    analyzer = LuggageAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()