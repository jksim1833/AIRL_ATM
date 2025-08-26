import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class LuggageAnalyzer:
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        
        # API 키 로드 (모델 초기화 전에 먼저 실행)
        self._load_api_keys()
        
        # 정확한 경로 설정 (Desktop 경로 사용)
        desktop_path = Path.home() / "Desktop"
        self.base_path = desktop_path / "AIRL_ATM" / "SW" / "chain1"
        self.prompt_path = self.base_path / "chain1_prompt" / "chain1_prompt.txt"
        self.image_path = self.base_path / "chain1_image" / f"{scenario_name}.jpeg"
        self.output_path = self.base_path / "chain1_out" / f"{scenario_name}.txt"
        
        # 프롬프트 로드
        self.system_prompt = self._load_prompt()
        
        # API 키 로드 후 모델 초기화
        print("🤖 모델 초기화 중...")
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        print("✅ 모델 초기화 완료")
    
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
            os.environ["GOOGLE_API_KEY"] = secrets["google"]["GOOGLE_API_KEY"]
            
            print("🔑 API 키 로드 완료")
            
        except KeyError as e:
            raise KeyError(f"🚨 API 키를 찾을 수 없습니다: {e}")
        except json.JSONDecodeError:
            raise ValueError("🚨 tetris_secrets.json 파일 형식이 올바르지 않습니다")
        
    def _load_prompt(self) -> str:
        """프롬프트 파일 로드 및 중괄호 이스케이프 처리"""
        try:
            with open(self.prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # JSON 예시의 중괄호들을 이스케이프 처리
            # 단일 중괄호 {를 {{로, }를 }}로 변경 (단, 이미 이스케이프된 것은 제외)
            import re
            
            # 이미 이스케이프된 {{ 와 }} 를 임시로 치환
            content = content.replace('{{', '__TEMP_OPEN__')
            content = content.replace('}}', '__TEMP_CLOSE__')
            
            # 나머지 단일 { } 를 {{ }} 로 변경
            content = content.replace('{', '{{')
            content = content.replace('}', '}}')
            
            # 임시 치환했던 것들을 원래대로 복원
            content = content.replace('__TEMP_OPEN__', '{{')
            content = content.replace('__TEMP_CLOSE__', '}}')
            
            print("🔧 프롬프트 중괄호 이스케이프 처리 완료")
            return content
            
        except FileNotFoundError:
            raise FileNotFoundError(f"🚨 프롬프트 파일을 찾을 수 없습니다: {self.prompt_path}")
    
    def _encode_image(self, image_path: Path) -> str:
        """이미지를 base64로 인코딩 (원본 크기/품질 유지)"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_chain(self):
        """LangChain 체인 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}
            ])
        ])
        
        chain = prompt | self.model
        
        return chain
    
    def analyze_image(self) -> str:
        """이미지 분석"""
        try:
            # 이미지 파일 존재 확인
            if not self.image_path.exists():
                raise FileNotFoundError(f"🚨 이미지 파일을 찾을 수 없습니다: {self.image_path}")
            
            # 체인 생성
            chain = self._create_chain()
            
            # 이미지 인코딩 (원본 그대로)
            image_data = self._encode_image(self.image_path)
            
            # 분석 실행
            result = chain.invoke({"image_data": image_data})
            
            # AIMessage에서 content 추출
            if hasattr(result, 'content'):
                return result.content
            else:
                return str(result)
                
        except Exception as e:
            print(f"❌ 분석 오류: {e}")
            return None
    
    def save_result(self, result: str):
        """결과 저장"""
        try:
            # 출력 디렉토리 생성
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 저장
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            print(f"💾 저장 완료: {self.output_path}")
            
        except Exception as e:
            print(f"❌ 저장 오류: {e}")
    
    def run_analysis(self):
        """전체 분석 실행"""
        print(f"🚀 === {self.scenario_name} 시나리오 분석 시작 ===")
        
        print(f"📸 분석할 이미지: {self.image_path}")
        print(f"📝 프롬프트 파일: {self.prompt_path}")
        print(f"💾 출력 경로: {self.output_path}")
        
        # 분석 실행
        result = self.analyze_image()
        
        if result:
            # 결과 저장
            self.save_result(result)
            print("✅ 분석 및 저장 완료")
        else:
            print("❌ 분석 실패")
        
        print(f"\n🎉 === {self.scenario_name} 시나리오 분석 완료 ===")

# 실행 함수
def main():
    """메인 실행 함수"""
    # 시나리오명 입력 받기
    scenario_name = input("시나리오명을 입력하세요: ")
    
    if not scenario_name.strip():
        print("❌ 시나리오명을 입력해주세요.")
        return
    
    analyzer = LuggageAnalyzer(scenario_name.strip())
    analyzer.run_analysis()

if __name__ == "__main__":
    main()