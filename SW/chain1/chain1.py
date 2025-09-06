import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time  # ⏱ 추가: 추론 시간 측정용

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class LuggageAnalyzer:
    def __init__(self, scenario_name: str, people_count: Optional[int] = None):
        self.scenario_name = scenario_name
        self.people_count = people_count  # ← 인원수 보관(후처리에서 JSON에 주입)

        # API 키 로드 (모델 초기화 전에 먼저 실행)
        self._load_api_keys()
        
        # 정확한 경로 설정 (Desktop 경로 사용)
        desktop_path = Path.home() / "Desktop"
        self.base_path = desktop_path / "AIRL_ATM" / "SW" / "chain1"
        self.prompt_path = self.base_path / "chain1_prompt" / "chain1_prompt_ver2_plus.txt"
        self.image_path = self.base_path / "chain1_image" / f"{scenario_name}.jpg"
        self.output_path = self.base_path / "chain1_out" / f"{scenario_name}.txt"
        
        # 프롬프트 로드
        self.system_prompt = self._load_prompt()
        
        # API 키 로드 후 모델 초기화
        print("🤖 모델 초기화 중...")
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
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
            
            # ⏱ 추가: 모델 호출~응답 시간 측정
            t0 = time.perf_counter()
            result = chain.invoke({"image_data": image_data})
            elapsed = time.perf_counter() - t0
            print(f"⏱️ 모델 호출~응답 시간: {elapsed:.3f}초")
            
            # AIMessage에서 content 추출
            if hasattr(result, 'content'):
                return result.content
            else:
                return str(result)
                
        except Exception as e:
            print(f"❌ 분석 오류: {e}")
            return None

    # ▼▼▼ 추가: LLM JSON 결과에 people을 주입하는 후처리 ▼▼▼
    def _inject_people(self, result_text: str) -> str:
        """
        LLM의 JSON 출력 문자열에 최상위 키로 {"people": <int>}를 주입하여
        { "people": N, ... } 형태로 반환한다.
        파싱 실패 시 {"people": N, "raw_model_output": "..."}로 안전 저장.
        """
        import re

        text = (result_text or "").strip()

        # ```json ... ``` 코드블록 처리
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
        if m:
            text = m.group(1).strip()

        # 주변 텍스트가 있으면 가장 바깥 { ... }만 추출
        if not (text.startswith("{") and text.endswith("}")):
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and first < last:
                text = text[first:last+1]

        try:
            data = json.loads(text)

            # people을 맨 앞에 두고 나머지 키를 이어 붙임(파이썬 dict는 삽입 순서 보장)
            out = {"people": int(self.people_count or 0)}
            if isinstance(data, dict):
                out.update(data)
            else:
                out["model_output"] = data  # dict가 아니면 원본 보존

            return json.dumps(out, ensure_ascii=False, indent=2)

        except Exception:
            fallback = {
                "people": int(self.people_count or 0),
                "raw_model_output": result_text
            }
            return json.dumps(fallback, ensure_ascii=False, indent=2)
    # ▲▲▲ 추가 끝 ▲▲▲
    
    def save_result(self, result: str):
        """결과 저장"""
        try:
            # 출력 디렉토리 생성
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JSON 문자열 그대로 저장 (people 주입 완료본)
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
        print(f"👥 인원수: {self.people_count}명")
        
        # 분석 실행
        result = self.analyze_image()
        
        if result:
            # 저장 전: LLM JSON에 people 주입
            result = self._inject_people(result)
            self.save_result(result)
            print("✅ 분석 및 저장 완료")
        else:
            print("❌ 분석 실패")
        
        print(f"\n🎉 === {self.scenario_name} 시나리오 분석 완료 ===")

# 실행 함수
def main():
    # 시나리오명 입력 받기
    scenario_name = input("시나리오명을 입력하세요: ")
    if not scenario_name.strip():
        print("❌ 시나리오명을 입력해주세요.")
        return

    # 인원수 입력 + 확인
    while True:
        ppl = input("차량 탑승 인원을 알려주세요! : ").strip()
        try:
            n = int(ppl)
            if n <= 0:
                print("❌ 양수로 입력하세요.")
                continue

            # 확인 단계
            confirmed = False
            while True:
                confirm = input(f"차량 탑승 인원은 \"{n}명\"이 맞나요? (1) 네 (2) 아니요 : ").strip()
                if confirm == "1":
                    people_count = n
                    confirmed = True
                    break
                elif confirm == "2": 
                    # 다시 처음부터 인원수 재입력
                    break
                else:
                    print("❌ 1 또는 2로 입력해주세요.")
            if confirmed:
                break  # 외부 루프 탈출(인원수 확정)

        except ValueError:
            print("❌ 숫자만 입력해주세요.")
            continue

    analyzer = LuggageAnalyzer(scenario_name.strip(), people_count=people_count)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
