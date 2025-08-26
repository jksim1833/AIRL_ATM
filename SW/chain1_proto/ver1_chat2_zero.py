import os
import json
import base64
from pathlib import Path
from PIL import Image
from langchain_openai import ChatOpenAI
import google.generativeai as genai
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, Tuple
import re

class LuggageVolumeEstimator:
    def __init__(self):
        # API 키 로드
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
        self.output_base = self.base_path / "chain1_out" / "chat2" / "zero"
        
        # 시나리오 폴더 목록
        self.scenarios = ["items_combined", "items_separated"]
        
        # 프롬프트 파일 로드
        self.zero_chat1_prompt = self._load_prompt("zero_chat1.txt")
        self.zero_chat2_prompt = self._load_prompt("zero_chat2.txt")
        
        # 부피 참조 객체들 (이름만)
        self.volume_references = {
            1: "축구공",
            2: "2L 생수 6개 묶음",
            3: "기내용 캐리어",
            4: "중형 캐리어", 
            5: "대형 캐리어",
            6: "드럼 세탁기"
        }
        
        # 랭체인 체인 설정
        self._setup_chains()

    def _load_prompt(self, filename: str) -> str:
        """프롬프트 파일 로드"""
        prompt_file = self.prompt_path / filename
        if not prompt_file.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _setup_chains(self):
        """랭체인 체인 설정"""
        # OpenAI 체인용 프롬프트 템플릿
        self.openai_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in luggage volume estimation and analysis."),
            ("human", [
                {"type": "text", "text": "{prompt_text}"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}
            ])
        ])
        
        # 출력 파서
        self.output_parser = StrOutputParser()

    def _create_openai_chain(self, model_name: str):
        """OpenAI 체인 생성"""
        llm = ChatOpenAI(
            model=model_name,
            api_key=self.openai_api_key,
            temperature=0.7,
            max_tokens=10000
        )
        return self.openai_prompt_template | llm | self.output_parser

    def _create_gemini_model(self, model_name: str):
        """Gemini 모델 생성"""
        return genai.GenerativeModel(
            model_name,
            system_instruction="You are an expert in luggage volume estimation and analysis.",
            generation_config=genai.types.GenerationConfig(temperature=0.3)
        )

    def encode_image(self, image_path: Path) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        image = Image.open(image_path)
        print(f"📷 이미지 크기: {image.size}")
        return base64.b64encode(image_data).decode('utf-8')

    def call_llm_with_history(self, model_name: str, image_path: Path, messages: list) -> str:
        """LLM 호출 (대화 히스토리 포함)"""
        try:
            if model_name.startswith("gpt"):
                # OpenAI 모델 - 메시지 히스토리 직접 전달
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=self.openai_api_key,
                    temperature=0.7,
                    max_tokens=10000
                )
                
                response = llm.invoke(messages)
                return response.content
                
            else:
                # Gemini 모델 - 히스토리를 텍스트로 변환하여 전달
                image = Image.open(image_path)
                
                # 메시지 히스토리를 텍스트로 변환
                history_text = ""
                current_prompt = ""
                
                for msg in messages:
                    if hasattr(msg, 'content'):
                        if isinstance(msg.content, list):
                            # 이미지가 포함된 메시지 처리
                            for part in msg.content:
                                if part.get('type') == 'text':
                                    if msg.__class__.__name__ == 'HumanMessage':
                                        current_prompt = part['text']
                                    elif msg.__class__.__name__ == 'AIMessage':
                                        history_text += f"Human: {current_prompt}\n\nAssistant: {part['text']}\n\n"
                        else:
                            # 텍스트만 있는 메시지
                            if msg.__class__.__name__ == 'HumanMessage':
                                current_prompt = msg.content
                            elif msg.__class__.__name__ == 'AIMessage':
                                history_text += f"Human: {current_prompt}\n\nAssistant: {msg.content}\n\n"
                
                # 현재 프롬프트를 히스토리에 추가
                final_prompt = history_text + f"Human: {current_prompt}"
                
                model = self._create_gemini_model(model_name)
                response = model.generate_content([
                    {"role": "user", "parts": [{"text": final_prompt}, image]}
                ])
                return response.text
                
        except Exception as e:
            print(f"⚠️ {model_name} 호출 실패: {e}")
            return None

    def get_user_volume_choice(self, luggage_description: str) -> Tuple[int, str]:
        """사용자로부터 부피 선택 받기"""
        print("\n다음 중에서 해당 짐과 부피가 가장 유사한 항목을 선택하세요:")
        print()
        
        # 후보 리스트 표시
        for i, name in self.volume_references.items():
            print(f"{i}. {name}")
        
        while True:
            try:
                choice = int(input("\n선택하세요 (1-5): "))
                if choice in self.volume_references:
                    selected_name = self.volume_references[choice]
                    print(f"\n✨ 선택됨: {choice}번 - {selected_name}")
                    
                    # 사용자 제공 정보 텍스트 생성
                    candidate_list = [f"{i}. {name}" for i, name in self.volume_references.items()]
                    candidate_list_text = "  ".join(candidate_list)
                    user_info_text = f"""후보 리스트:
{candidate_list_text}
응답: {choice}번 ({selected_name})"""
                    
                    return choice, user_info_text
                else:
                    print("⚠️ 1-5 사이의 숫자를 입력해주세요.")
            except ValueError:
                print("⚠️ 숫자를 입력해주세요.")

    def create_reasoning_prompt(self, user_choice: int) -> str:
        """Reasoning Extraction 프롬프트 생성 (사용자 정보 + zero_chat2.txt만)"""
        selected_name = self.volume_references[user_choice]
        
        # 후보 리스트 생성
        candidate_list = [f"{i}. {name}" for i, name in self.volume_references.items()]
        candidate_list_text = "  ".join(candidate_list)
        
        user_info_section = f"""[정보 제공] 다음 후보 중 해당 짐과 부피가 가장 유사한 항목은 다음과 같습니다.

후보 리스트:
{candidate_list_text}
응답: {user_choice}번 ({selected_name})

"""
        
        return user_info_section + self.zero_chat2_prompt
    
    def create_answer_prompt(self, reasoning_prompt: str, reasoning_response: str) -> str:
        """Answer Extraction 프롬프트 생성 (Reasoning + 응답 + "그러므로 답은:")"""
        return f"""{reasoning_prompt}

{reasoning_response}

그러므로 답은: (전체 부피를 리터 단위 숫자로만 출력하세요. 예: 94)"""

    def save_final_results(self, model_name: str, scenario: str, image_name: str, 
                          step1_response: str, user_info: str, 
                          reasoning_response: str, final_response: str):
        """최종 결과 파일 저장"""
        output_dir = self.output_base / model_name / scenario
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 섹션 헤더만 하드코딩, 내용은 동적
        final_content = f"""[선택된 짐]
{step1_response}

[사용자 제공 정보]
{user_info}

[전체 부피 추론 과정]
{reasoning_response}

[최종 추정 전체 부피]
{final_response}"""
        
        output_path = output_dir / f"{image_name}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        print(f"📁 최종 결과 저장: {output_path}")

    def process_single_image(self, model_name: str, scenario: str, image_name: str):
        """단일 이미지 처리"""
        print(f"\n{'='*80}")
        print(f"처리 중: {model_name} | {scenario} | {image_name}")
        print(f"{'='*80}")
        
        image_path = self.image_base_path / scenario / f"{image_name}.jpeg"
        
        if not image_path.exists():
            print(f"❌ 이미지를 찾을 수 없습니다: {image_path}")
            return
        
        base64_image = self.encode_image(image_path)
        
        # Step 1: zero_chat1.txt 프롬프트로 짐 선택
        print("\n=== Step 1: 부피 분석하기 쉬운 짐 선택 ===")
        
        # 초기 메시지 구성 (이미지 + zero_chat1 프롬프트)
        system_msg = SystemMessage(content="You are an expert in luggage volume estimation and analysis.")
        step1_user_msg = HumanMessage(content=[
            {"type": "text", "text": self.zero_chat1_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])
        
        step1_messages = [system_msg, step1_user_msg]
        step1_response = self.call_llm_with_history(model_name, image_path, step1_messages)
        
        if not step1_response:
            print("❌ Step 1 실패")
            return
        
        print("🚀 LLM 응답:")
        print(step1_response)
        
        # Step 2: 사용자 부피 선택
        print("\n=== Step 2: 사용자 부피 선택 ===")
        user_choice, user_info_text = self.get_user_volume_choice(step1_response)
        
        # Step 3: Reasoning Extraction (히스토리 + 사용자 정보 + zero_chat2.txt)
        print("\n=== Step 3: Reasoning Extraction ===")
        
        # 히스토리에 Step 1 응답 추가
        step1_ai_msg = AIMessage(content=step1_response)
        
        # Reasoning Extraction 프롬프트 생성 (사용자 정보 + zero_chat2.txt만)
        reasoning_prompt = self.create_reasoning_prompt(user_choice)
        step3_user_msg = HumanMessage(content=[
            {"type": "text", "text": reasoning_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])
        
        # 히스토리 포함한 메시지 구성
        step3_messages = [system_msg, step1_user_msg, step1_ai_msg, step3_user_msg]
        reasoning_response = self.call_llm_with_history(model_name, image_path, step3_messages)
        
        if not reasoning_response:
            print("❌ Step 3 (Reasoning Extraction) 실패")
            return
        
        print("🔍 추론 과정:")
        print(reasoning_response)
        
        # Step 4: Answer Extraction (전체 히스토리 + Reasoning Extraction + Step3응답 + "그러므로 답은:")
        print("\n=== Step 4: Answer Extraction ===")
        
        # Answer Extraction 프롬프트 = Reasoning Extraction + Step3응답 + "그러므로 답은:"
        answer_prompt = f"""{reasoning_prompt}

{reasoning_response}

그러므로 답은:"""
        
        step4_user_msg = HumanMessage(content=[
            {"type": "text", "text": answer_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])
        
        # 히스토리는 이미지 + zero_chat1 + Step1응답만
        step4_messages = [system_msg, step1_user_msg, step1_ai_msg, step4_user_msg]
        step4_response = self.call_llm_with_history(model_name, image_path, step4_messages)
        
        if not step4_response:
            print("❌ Step 4 (Answer Extraction) 실패")
            return
        
        print("✅ 최종 답변:")
        print(step4_response)
        
        # 최종 결과 파일 저장
        self.save_final_results(
            model_name, scenario, image_name,
            step1_response, user_info_text, 
            reasoning_response, step4_response
        )
        
        print("🎉 처리 완료!")

    def process_all_images(self):
        """모든 이미지 처리"""
        print("🎯 짐 부피 추정 시스템 시작...")
        
        # 프롬프트 파일 존재 확인
        required_files = ["zero_chat1.txt", "zero_chat2.txt"]
        for filename in required_files:
            file_path = self.prompt_path / filename
            if not file_path.exists():
                raise FileNotFoundError(f"필수 프롬프트 파일을 찾을 수 없습니다: {file_path}")
        
        for model_name in self.models:
            print(f"\n{'='*100}")
            print(f"모델: {model_name} 처리 시작")
            print(f"{'='*100}")
            
            for scenario in self.scenarios:
                print(f"\n시나리오: {scenario}")
                
                for image_name in ["1", "2"]:
                    try:
                        self.process_single_image(model_name, scenario, image_name)
                        
                        # 다음 이미지 처리 전 확인
                        user_input = input("\n다음 이미지로 계속하시겠습니까? (Enter: 계속, q: 종료): ")
                        if user_input.lower() == 'q':
                            print("⏹️ 처리 중단됨")
                            return
                            
                    except KeyboardInterrupt:
                        print("\n\n⏹️ 사용자에 의해 중단됨")
                        return
                    except Exception as e:
                        print(f"❌ 이미지 처리 실패 ({model_name}/{scenario}/{image_name}): {e}")
                        continue
        
        print("\n" + "="*100)
        print("🏁 모든 처리 완료!")
        print("="*100)

# 실행 코드
if __name__ == "__main__":
    estimator = LuggageVolumeEstimator()
    estimator.process_all_images()